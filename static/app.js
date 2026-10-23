// Nova Sales Assistant: front end. Talks to the FastAPI backend under /api.

const state = {
  leads: [],
  priority: "",
  search: "",
  selectedId: null,
  selected: null,
  sessionId: null,  // one chat per page load; the server keeps its memory
  busy: false,
};

const SUGGESTIONS = [
  "What service are they interested in?",
  "Is their budget enough for it?",
  "They said the price is too high. How should I respond?",
  "What discount can I give without approval?",
  "What are the payment terms for projects?",
];

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), search: $("search"), count: $("lead-count"), list: $("lead-list"),
  card: $("lead-card"), messages: $("messages"), suggestions: $("suggestions"),
  composer: $("composer"), input: $("input"), send: $("send"),
  clear: $("clear"), memoryToggle: $("memory-toggle"), memory: $("memory"),
};

// ---------- helpers ----------

function saveSession(id) {
  state.sessionId = id;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Model answers can contain **bold** and line breaks; everything else stays plain text.
function formatAnswer(text) {
  return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.status === 204 ? null : response.json();
}

// ---------- status ----------

async function loadHealth() {
  try {
    const health = await api("/api/health");
    els.status.textContent = health.mock ? "Mock mode: no model loaded" : `Ready with ${health.model.split("/").pop()}`;
    els.status.className = "status " + (health.mock ? "mock" : "ok");
  } catch (error) {
    els.status.textContent = "Server not ready: " + error.message;
    els.status.className = "status error";
  }
}

// ---------- lead list ----------

async function loadLeads() {
  try {
    state.leads = await api("/api/leads");
    renderList();
  } catch (error) {
    els.count.textContent = "Leads could not be loaded: " + error.message;
  }
}

function renderList() {
  const search = state.search.toLowerCase();
  const visible = state.leads.filter((lead) =>
    (!state.priority || lead.priority === state.priority) &&
    (!search || lead.company.toLowerCase().includes(search)));

  els.count.textContent = `${visible.length} of ${state.leads.length} leads`;
  els.list.innerHTML = visible.map((lead) => `
    <li>
      <button type="button" class="bar-${lead.priority.toLowerCase()}" data-id="${escapeHtml(lead.lead_id)}"
              aria-current="${lead.lead_id === state.selectedId}">
        <span class="name">${escapeHtml(lead.company)}${lead.at_risk ? '<span class="risk-dot">At risk</span>' : ""}</span>
        <span class="meta">${escapeHtml(lead.service)}, ${escapeHtml(lead.stage)}</span>
      </button>
    </li>`).join("");
}

els.list.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-id]");
  if (button) selectLead(button.dataset.id);
});

els.search.addEventListener("input", () => { state.search = els.search.value; renderList(); });

document.querySelectorAll(".filters button").forEach((button) => {
  button.addEventListener("click", () => {
    state.priority = button.dataset.priority;
    document.querySelectorAll(".filters button").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    renderList();
  });
});

// ---------- lead card ----------

async function selectLead(leadId) {
  if (!leadId) return;
  state.selectedId = leadId;
  renderList();
  try {
    state.selected = await api(`/api/leads/${encodeURIComponent(leadId)}`);
    renderCard(state.selected);
  } catch (error) {
    els.card.className = "lead-card empty";
    els.card.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
  }
}

function renderCard(lead) {
  const check = lead.check;
  const flags = [];
  if (check.at_risk) flags.push(`<li class="flag risk">No contact for ${check.days_since_contact} days. Contact them today.</li>`);
  check.reasons.forEach((reason) => flags.push(`<li class="flag">${escapeHtml(reason)}</li>`));
  if (check.special_discount) flags.push(`<li class="flag good">${escapeHtml(lead.industry)} lead: eligible for ${check.special_discount}% off.</li>`);

  els.card.className = "lead-card";
  els.card.innerHTML = `
    <div class="card-head">
      <div>
        <h2>${escapeHtml(lead.company)}</h2>
        <p class="sub">${escapeHtml(lead.industry)} in ${escapeHtml(lead.city)}, ${lead.employees} employees</p>
      </div>
      <span class="chip chip-${lead.priority.toLowerCase()}">${escapeHtml(lead.priority)}</span>
    </div>
    <dl class="facts">
      <div><dt>Contact</dt><dd>${escapeHtml(lead.contact_person)}, ${escapeHtml(lead.job_title)}</dd></div>
      <div><dt>Wants</dt><dd>${escapeHtml(lead.service)} (${escapeHtml(check.price)})</dd></div>
      <div><dt>Budget</dt><dd>$${lead.budget.toLocaleString("en-US")} ${escapeHtml(lead.budget_type.toLowerCase())}</dd></div>
      <div><dt>Stage</dt><dd>${escapeHtml(lead.stage)}</dd></div>
      <div><dt>Next step</dt><dd>${escapeHtml(check.next_action)}</dd></div>
      <div><dt>Source</dt><dd>${escapeHtml(lead.lead_source)}, last contact ${escapeHtml(lead.last_contact)}</dd></div>
    </dl>
    ${flags.length ? `<ul class="flags">${flags.join("")}</ul>` : ""}
    <div class="card-foot">
      <p class="notes">${escapeHtml(lead.notes)}</p>
      <button type="button" class="primary" id="brief">Brief me before the call</button>
    </div>`;
  $("brief").addEventListener("click", () => send(`Give me a short brief about ${lead.company} before my call.`));
}

// ---------- chat ----------

function addMessage(role, html) {
  els.messages.querySelector(".empty-chat")?.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = html;
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  const brief = $("brief");
  if (brief) brief.disabled = busy;
}

async function send(text) {
  text = text.trim();
  if (!text || state.busy) return;
  setBusy(true);
  addMessage("user", escapeHtml(text));
  const thinking = addMessage("assistant thinking", "Thinking");

  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message: text, session_id: state.sessionId, lead_id: state.selectedId }),
    });
    saveSession(result.session_id);
    thinking.className = "msg assistant";
    thinking.innerHTML = `${formatAnswer(result.answer)}<span class="meta">${(result.latency_ms / 1000).toFixed(1)} s</span>`;
    if (result.lead_id && result.lead_id !== state.selectedId) await selectLead(result.lead_id);
    if (!els.memory.hidden) renderMemory(result.memory);
  } catch (error) {
    thinking.className = "msg error";
    thinking.textContent = `No answer: ${error.message}`;
  } finally {
    setBusy(false);
    els.messages.scrollTop = els.messages.scrollHeight;
  }
}

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = els.input.value;
  els.input.value = "";
  autosize();
  send(text);
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});

function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = els.input.scrollHeight + "px";
}
els.input.addEventListener("input", autosize);

els.suggestions.innerHTML = SUGGESTIONS.map((q) => `<button type="button">${escapeHtml(q)}</button>`).join("");
els.suggestions.addEventListener("click", (event) => {
  if (event.target.tagName === "BUTTON") send(event.target.textContent);
});

// ---------- memory ----------

function renderMemory(memory) {
  els.memory.textContent =
    `Turns kept in full: ${memory.turns_in_full}\n\nSummary of older turns:\n` +
    (memory.summary || "Nothing summarized yet. After 5 turns, the oldest 3 are summarized.");
}

els.memoryToggle.addEventListener("click", async () => {
  const open = els.memory.hidden;
  els.memory.hidden = !open;
  els.memoryToggle.setAttribute("aria-expanded", String(open));
  if (open) {
    const memory = state.sessionId
      ? await api(`/api/sessions/${state.sessionId}/memory`).catch(() => ({ turns_in_full: 0, summary: "" }))
      : { turns_in_full: 0, summary: "" };
    renderMemory(memory);
  }
});

els.clear.addEventListener("click", async () => {
  if (state.sessionId) await api(`/api/sessions/${state.sessionId}`, { method: "DELETE" }).catch(() => {});
  saveSession(null);
  els.messages.innerHTML = '<div class="empty-chat"><p>Chat and memory cleared. Pick a lead or ask a question.</p></div>';
  if (!els.memory.hidden) renderMemory({ turns_in_full: 0, summary: "" });
});

// ---------- start ----------

loadHealth();
loadLeads();
