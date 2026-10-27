# Demo script

A 5-minute walk through the project, and the questions people usually ask.

## Before the demo

1. Start the app 15 minutes early (Colab: **Run all**; the model takes 5 to 10 minutes to load the first time). Keep the Colab tab open.
2. Open the web app and check the top right says **Ready with Qwen2.5-7B-Instruct**.
3. Ask one warm-up question so the model is warm, then press **Clear chat and memory**.
4. Have a backup: the screenshots in `screenshots/`, or run on your laptop with Ollama (see the README). If the GPU is not available, `MOCK_LLM=1` still shows the whole app and the CRM rules, only the answers are not written by a model.

## The flow

| Step | Do | Point out |
|---|---|---|
| 1 | Click **Hot**, then **At risk** in the lead list | Priority and at-risk are calculated in Python from the playbook rules, not by the model. |
| 2 | Open **Nile Pharma** | The card: budget vs. price, decision maker, next step. All rules, no LLM. |
| 3 | Press **Brief me before the call** | A brief built from the CRM row + the playbook. Open **Sources** under the answer. |
| 4 | Ask *"Is their budget enough for it?"* | Memory: "their" and "it" are understood. |
| 5 | Ask *"They said the price is too high. How should I respond?"* | Objection handling from the playbook: a cheaper package with its real features and price, the 50/30/20 split. |
| 6 | Open **Sinai Tours** and ask *"Is their budget enough?"* | $4,000 for a $25,000 app: the assistant says no, and that no app package fits that budget. |
| 7 | Ask *"Compare Nile Pharma and Sinai Tours"* | Two leads in one question. |
| 8 | Ask *"How many hot leads do we have?"* | Counts come from Python (the CRM overview), the model never counts rows. |
| 9 | Ask *"What is the capital of France?"* | "I don't have this information." The model only answers from the data. |
| 10 | Open **What the assistant remembers** after 5 questions | Summary memory: the oldest 3 turns are summarized. |
| 11 | Open `/docs` | The same API the web app uses, with live docs. |

## Questions people ask

**How do you stop the model from making things up?**
Five layers. (1) Retrieval: when the lead is known, the model gets that exact CRM row and only playbook chunks, so other companies cannot leak in. (2) The rules (budget, decision maker, priority, at risk) are calculated in Python and given to the model as a "CRM check" it must trust. (3) The prompt forbids outside knowledge and gives a fixed sentence for unknown answers. (4) Greedy decoding, so the same question gives the same answer. (5) A fact guard checks every price and percentage in the answer against the data, asks the model again if one is made up, and flags it in the UI if it is still there.

**Why two FAISS indexes?**
When we already know which lead the rep means, searching the lead rows again would return other companies with similar text, and the model could mix them up. So we take the exact row and search the playbook only.

**Why are the rules in Python and not in the prompt?**
Comparing $4,000 with $25,000 or counting days since contact is easy for code and easy to get wrong for a language model. Code is exact and can be tested (`tests/test_rules.py`).

**How does memory work?**
Each browser gets a session id. The server keeps the last turns in full; after 5 turns, the oldest 3 are summarized by the model. The lead is found in this order: named in the question, open in the UI, named earlier in the chat, named in the summary.

**Why Qwen2.5-7B?**
Good at following instructions, free, and in 4-bit it fits on a free Colab T4 (about 5 GB of GPU memory). The code also works with any OpenAI-compatible server, so it can run with Ollama on a laptop or a hosted model.

**Why is "at risk" counted from 22 October 2026?**
The CRM sheet is a snapshot from that date. Counting from the real date would make every lead look "at risk" a month later. With live data, `AS_OF_DATE=today`.

**How did you evaluate it?**
In the notebook: 14 factual questions and 4 follow-ups, all correct, and a question outside the data. Plus 88 automated tests for the rules, lead matching, memory, the fact guard and the API, run by GitHub Actions on every push. The honest limitation: 18 questions is small; the next step is a larger set with an LLM-as-judge.

**What would you add for production?**
Login so each rep sees only their leads, Redis for sessions, vLLM for serving many users, and connecting to the real CRM instead of an Excel sheet.

## If something goes wrong

| Problem | Fix |
|---|---|
| Colab: "No GPU" | Runtime → Change runtime type → T4 GPU, then Run all. |
| The link does not open | Wait 20 seconds; if still not, run step 3 again for a new link. |
| "The model is still loading" | Wait a minute; the first start downloads the model. |
| An answer shows **Check: ...** | That is the fact guard working: say so, it caught a number that is not in the data. |
| Slow answers | The first answer after loading is slower; later ones take 3 to 10 s on a T4. |
