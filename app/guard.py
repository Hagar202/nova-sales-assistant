"""
Fact guard: catch prices and percentages the model made up.

Every dollar amount and percentage in an answer must come from our data (the playbook,
the retrieved CRM rows, the question) or be simple math on those numbers: a discount,
the 50/30/20 payment split (also after a discount), a few months of a monthly price,
a setup fee plus monthly fees, or a budget gap. If the model writes a number that
cannot be traced back, the assistant asks it to answer again, and if the number is
still there, the answer is flagged in the UI.

It also checks prices next to package names: "the Basic App costs $5,000" is caught
even though $5,000 is a real price (of the AI Chatbot), because the Basic App costs $6,000.
"""

import re

_AMOUNT = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_MONEY = re.compile(rf"(?:US\$|\$|USD\s?)\s?(?P<a>{_AMOUNT})\s*(?P<k>k|K|thousand)?\b"
                    rf"|(?P<b>{_AMOUNT})\s*(?:USD|dollars)\b")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|percent\b)")
_NUMBER = re.compile(r"(?<![\w.,])" + _AMOUNT + r"(?![\w])")

# Month counts used in contracts (3- and 6-month minimums, a year), for monthly amounts only.
MULTIPLIERS = (2, 3, 6, 12)
# Monthly prices, hosting, ad budgets, hourly rates and bonuses are all at or below this ($1,200 = Ads Management).
SMALL_AMOUNT = 1200
# Project payments: 50% at signing, 30% in the middle, 20% at delivery.
PAYMENT_SPLIT = (50, 30, 20)


def _to_float(text):
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def money_in(text):
    """Dollar amounts written in the text, e.g. "$25,000", "$1.5k", "4000 USD"."""
    amounts = []
    for match in _MONEY.finditer(text):
        raw = match.group("a") or match.group("b")
        value = _to_float(raw)
        if value is None:
            continue
        if match.group("k"):
            value *= 1000
        amounts.append((match.group(0).strip().rstrip(","), value))
    return amounts


def percents_in(text):
    return [(m.group(0).strip(), float(m.group(1))) for m in _PERCENT.finditer(text)]


def plain_numbers_in(text):
    return {v for v in (_to_float(n) for n in _NUMBER.findall(text)) if v is not None}


def allowed_amounts(amounts, percents, budgets=()):
    """
    Amounts the model may write: the known ones, and simple math on them.
    amounts: prices and other amounts from the data. budgets: the lead's own amounts (for budget gaps).
    """
    amounts = {a for a in amounts if a > 0}
    small = {a for a in amounts if a <= SMALL_AMOUNT}
    monthly_totals = {s * m for s in small for m in MULTIPLIERS}

    discounted = set()      # a discount, the price after it, a payment part or a commission
    for a in amounts | monthly_totals:
        for p in percents:
            discounted.add(a * p / 100)
            discounted.add(a * (100 - p) / 100)

    allowed = amounts | monthly_totals | discounted
    # Second step: the 50/30/20 payment parts of a discounted price, and months of a discounted monthly price
    for d in discounted:
        allowed.update(d * p / 100 for p in PAYMENT_SPLIT)
        if d <= SMALL_AMOUNT:
            allowed.update(d * m for m in MULTIPLIERS)
    # Totals: a price plus monthly fees or an ad budget
    for a in amounts | monthly_totals:
        for b in small | monthly_totals:
            allowed.add(a + b)
    # Budget gaps: the lead's budget against a price
    for budget in budgets:
        allowed.update(abs(budget - a) for a in amounts)
    return allowed


def _close(value, candidates):
    tolerance = max(1.0, value * 0.0001)  # rounding to the dollar
    return any(abs(value - c) <= tolerance for c in candidates)


# How a package's price can legitimately appear: the price, a discount on it, a payment part,
# a few months of it (monthly packages), or months of a discounted monthly price.
def package_amounts(price, monthly, percents, extra_monthly=()):
    base = {price}
    monthly_parts = set(extra_monthly) | ({price} if monthly else set())
    for m in monthly_parts:
        base.update(m * k for k in MULTIPLIERS)
    if extra_monthly:  # setup fee + monthly fee (AI Chatbot: $5,000 + $300 / month)
        base.update(price + m * k for m in extra_monthly for k in (1,) + MULTIPLIERS)
        base.update(extra_monthly)
    allowed = set(base)
    for b in base:
        for p in (p for p in percents if p < 100):  # a package is never sold above its price
            after = b * (100 - p) / 100
            allowed.update({b * p / 100, after})
            allowed.update(after * q / 100 for q in PAYMENT_SPLIT)
    for b in list(allowed):
        allowed.update(b + s * k for s in (500,) for k in (1,) + MULTIPLIERS if monthly)  # ads + ad budget
    return allowed


class FactGuard:
    """Built once with the whole playbook, then checks each answer against it plus the retrieved context."""

    # Package name, then at most this many characters, then the price: "Basic App costs $6,000"
    PAIR_GAP = 30

    def __init__(self, playbook_text="", package_prices=None, monthly_packages=(), monthly_fees=None):
        self.base_amounts = {v for _, v in money_in(playbook_text)}
        self.base_percents = {v for _, v in percents_in(playbook_text)} | {100.0}
        self.package_prices = dict(package_prices or {})
        self.monthly_packages = set(monthly_packages)
        self.monthly_fees = dict(monthly_fees or {})
        names = sorted(self.package_prices, key=len, reverse=True)
        self._package_re = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I) if names else None
        self._canonical = {n.lower(): n for n in names}

    def wrong_package_prices(self, answer, percents):
        """Prices written right after a package name that do not match that package."""
        if not self._package_re:
            return []
        problems = []
        mentions = list(self._package_re.finditer(answer))
        for i, mention in enumerate(mentions):
            end = mentions[i + 1].start() if i + 1 < len(mentions) else len(answer)
            window = answer[mention.end():min(end, mention.end() + self.PAIR_GAP + 20)]
            money = next(_MONEY.finditer(window), None)
            if money is None or money.start() > self.PAIR_GAP:
                continue
            between = window[:money.start()].lower()
            if any(word in between for word in ("budget", ".", ";", " and ", " or ", "save", "short")):
                continue  # the amount is about something else
            value = money_in(money.group(0))[0][1]
            name = self._canonical[mention.group(1).lower()]
            allowed = package_amounts(self.package_prices[name], name in self.monthly_packages, percents,
                                      self.monthly_fees.get(name, ()))
            if not _close(value, allowed):
                problems.append(f"{money.group(0).strip().rstrip(',')} for the {name}")
        return problems

    def unsupported(self, answer, reference_text, user_text=""):
        """
        Dollar amounts and percentages in the answer that the data does not support.
        reference_text: retrieved context. user_text: what the sales rep wrote (plain numbers count too).
        """
        amounts = self.base_amounts | {v for _, v in money_in(reference_text + "\n" + user_text)}
        amounts |= {v for v in plain_numbers_in(user_text) if v >= 100}
        percents = self.base_percents | {v for _, v in percents_in(reference_text + "\n" + user_text)}
        budgets = amounts - self.base_amounts  # amounts that come from the CRM rows or the question
        allowed = allowed_amounts(amounts, percents, budgets)

        problems = self.wrong_package_prices(answer, percents)
        flagged = {p.split(" for the ")[0] for p in problems}
        for text, value in money_in(answer):
            if text not in flagged and not _close(value, allowed):
                problems.append(text)
        for text, value in percents_in(answer):
            if value not in percents:
                problems.append(text)
        return list(dict.fromkeys(problems))  # keep order, drop duplicates


def unsupported_numbers(answer, reference_text, user_text=""):
    """Shortcut without a playbook: only the reference text counts as data."""
    return FactGuard().unsupported(answer, reference_text, user_text)
