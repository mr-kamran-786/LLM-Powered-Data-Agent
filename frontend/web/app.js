/* Data Agent web UI: plain JavaScript, no build step. Talks to the FastAPI backend. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const state = { session: null, config: null, busy: false };

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const icon = {
  ok: '<svg viewBox="0 0 24 24"><path d="M5 12l5 5L20 7"/></svg>',
  err: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
};

function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), isError ? 6000 : 3000);
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {}, ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res;
}

const fmtNum = (v) => {
  if (typeof v !== "number") return v === null || v === undefined ? "" : String(v);
  if (Number.isInteger(v)) return v.toLocaleString("en-US");
  const a = Math.abs(v);
  return a >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 2 }) : Number(v.toPrecision(6)).toString();
};

/** Minimal, safe Markdown: escape first, then add formatting. */
function md(text) {
  const lines = esc(text || "").split("\n");
  const out = [];
  let list = null;
  let inCode = false;
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>");
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.trim().startsWith("```")) {
      if (list) { out.push(`</${list}>`); list = null; }
      out.push(inCode ? "</pre>" : '<pre class="out">');
      inCode = !inCode;
      continue;
    }
    if (inCode) { out.push(line); continue; }
    const ul = line.match(/^\s*[-*•]\s+(.*)/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)/);
    if (ul || ol) {
      const kind = ul ? "ul" : "ol";
      if (list !== kind) { if (list) out.push(`</${list}>`); out.push(`<${kind}>`); list = kind; }
      out.push(`<li>${inline((ul || ol)[1])}</li>`);
      continue;
    }
    if (list) { out.push(`</${list}>`); list = null; }
    const h = line.match(/^(#{1,4})\s+(.*)/);
    if (h) out.push(`<h4>${inline(h[2])}</h4>`);
    else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push(`</${list}>`);
  if (inCode) out.push("</pre>");
  return out.join("\n");
}

/** Light syntax highlighting for Python/SQL (escaped input). */
function highlight(code, lang) {
  const kw = lang === "sql"
    ? /\b(SELECT|FROM|WHERE|GROUP|BY|ORDER|LIMIT|JOIN|LEFT|RIGHT|INNER|OUTER|ON|AS|AND|OR|NOT|IN|IS|NULL|COUNT|SUM|AVG|MIN|MAX|DISTINCT|HAVING|WITH|CASE|WHEN|THEN|ELSE|END|DESC|ASC|UNION|ALL)\b/gi
    : /\b(import|from|as|def|return|if|elif|else|for|in|while|and|or|not|is|None|True|False|lambda|with|try|except|print)\b/g;
  const tokens = [];
  const stash = (cls) => (m) => { tokens.push(`<span class="${cls}">${m}</span>`); return `\u0000${tokens.length - 1}\u0000`; };
  let s = esc(code);
  s = s.replace(lang === "sql" ? /--[^\n]*/g : /#[^\n]*/g, stash("com"));
  s = s.replace(/(&quot;|&#39;)(?:(?!\1).)*\1/g, stash("str"));
  s = s.replace(kw, stash("kw"));
  s = s.replace(/\b\d+(\.\d+)?\b/g, stash("num"));
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => tokens[+i]);
}

function tableHTML(t, maxRows = 50) {
  if (!t || !t.columns) return "";
  const numeric = t.columns.map((_, j) => t.rows.slice(0, 20).every((r) => r[j] === null || typeof r[j] === "number"));
  const head = t.columns.map((c, j) => `<th class="${numeric[j] ? "n" : ""}">${esc(c)}</th>`).join("");
  const body = t.rows.slice(0, maxRows).map((r) => "<tr>" + r.map((v, j) => `<td class="${numeric[j] ? "n" : ""}">${esc(fmtNum(v))}</td>`).join("") + "</tr>").join("");
  const [nr, nc] = t.shape || [t.rows.length, t.columns.length];
  const cap = `${t.title ? `<strong>${esc(t.title)}</strong> · ` : ""}${nr.toLocaleString()} rows × ${nc} columns${t.truncated ? ` · showing first ${Math.min(maxRows, t.rows.length)}` : ""}`;
  return `<div><p class="tbl-cap">${cap}</p><div class="tbl-wrap"><table class="data"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div></div>`;
}

function renderChart(container, chart) {
  if (chart.kind === "image") {
    const img = document.createElement("img");
    img.className = "chart";
    img.alt = chart.title || "chart";
    img.src = `data:image/png;base64,${chart.png}`;
    container.appendChild(img);
    return;
  }
  const div = document.createElement("div");
  div.className = "chart";
  container.appendChild(div);
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const layout = Object.assign({}, chart.spec.layout || {}, {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", autosize: true,
    margin: { l: 60, r: 24, t: 50, b: 50 }, font: { color: dark ? "#e6e9ef" : "#1b2230", family: "Inter, system-ui, sans-serif" },
  });
  if (dark) delete layout.template;
  const draw = () => window.Plotly && Plotly.newPlot(div, chart.spec.data || [], layout, { responsive: true, displaylogo: false });
  if (window.Plotly) draw(); else window.addEventListener("load", draw, { once: true });
}

// ------------------------------------------------------------------ sessions
async function loadConfig() {
  state.config = await api("/api/config");
  const sel = $("#modelSelect");
  sel.innerHTML = state.config.models.map((m) => `<option value="${esc(m)}">${esc(m.replace(/^gemini\//, "Gemini · ").replace(/^ollama\//, "Ollama · "))}</option>`).join("");
  sel.value = localStorageGet("model") || state.config.default_model;
  if (!state.config.models.includes(sel.value)) sel.value = state.config.default_model;
  $("#criticToggle").checked = localStorageGet("critic") !== "0" && state.config.critic_enabled;
  $("#maxMb").textContent = state.config.max_upload_mb;
}

function localStorageGet(k) { try { return localStorage.getItem("da." + k); } catch (_) { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem("da." + k, v); } catch (_) { /* private mode */ } }

async function refreshSessionList() {
  const sessions = await api("/api/sessions");
  const ul = $("#sessionList");
  ul.innerHTML = sessions.map((s) => `<li data-id="${s.id}" class="${state.session?.id === s.id ? "active" : ""}">
      <span class="s-title" title="${esc(s.title)}">${esc(s.title)}</span>
      <button class="icon-btn del" data-del="${s.id}" aria-label="Delete">${icon.trash}</button></li>`).join("")
    || '<li class="muted" style="cursor:default">No analyses yet</li>';
}

async function openSession(id) {
  state.session = await api(`/api/sessions/${id}`);
  localStorageSet("session", id);
  renderSession();
  refreshSessionList();
}

async function newSession() {
  state.session = await api("/api/sessions", { method: "POST", body: JSON.stringify({ title: "New analysis" }) });
  localStorageSet("session", state.session.id);
  renderSession();
  refreshSessionList();
}

async function ensureSession() {
  if (!state.session) await newSession();
  return state.session;
}

function renderSession() {
  const s = state.session;
  $("#sessionTitle").textContent = s.title || "New analysis";
  renderDatasets();
  renderUsage();
  const thread = $("#thread");
  thread.innerHTML = "";
  if (!s.turns.length) thread.appendChild(emptyState());
  for (const t of s.turns) {
    thread.appendChild(userBubble(t.question));
    const card = newAgentCard();
    thread.appendChild(card.el);
    if (t.plan) card.plan(t.plan);
    for (const step of t.steps || []) card.step(step);
    if (t.status === "clarification") card.clarify(t.answer_md);
    else card.answer({ answer_md: t.answer_md, status: t.status, short_answer: t.short_answer });
    card.done(t.usage);
  }
  renderSuggestions();
  thread.scrollTop = thread.scrollHeight;
}

function emptyState() {
  const tpl = document.createElement("div");
  tpl.className = "empty";
  tpl.id = "emptyState";
  tpl.innerHTML = `<h2>Ask your data a question</h2>
    <p>Upload a file or connect a database, then ask in plain language. The agent plans the analysis,
    writes and runs the code in a sandbox, checks the result, and shows you every step.</p>
    <div class="howto"><div><span>1</span>Load data</div><div><span>2</span>Ask a question</div><div><span>3</span>Inspect code, tables &amp; charts</div></div>`;
  return tpl;
}

function renderUsage() {
  const u = state.session?.usage;
  if (!u) return;
  const tokens = u.tokens_in + u.tokens_out;
  $("#usage").textContent = `${(tokens / 1000).toFixed(1)}k tokens · $${u.cost_usd.toFixed(4)}`;
}

function renderDatasets() {
  const s = state.session;
  const items = s.datasets.map((d) => {
    const isDb = d.source.startsWith("db:");
    return `<li data-ds="${esc(d.name)}" title="Open profile">
      <span class="ds-icon ${isDb ? "db" : ""}">${isDb ? "SQL" : esc((d.filename || "").split(".").pop().toUpperCase().slice(0, 4))}</span>
      <span style="min-width:0"><div class="ds-name">${esc(d.name)}</div>
      <div class="ds-meta">${d.rows.toLocaleString()} rows · ${d.columns} cols${isDb ? " · " + esc(d.source.slice(3)) : ""}</div></span></li>`;
  });
  $("#datasetList").innerHTML = items.join("");
  $("#datasetCount").textContent = s.datasets.length ? `(${s.datasets.length})` : "";
}

function renderSuggestions() {
  const names = new Set(state.session?.datasets.map((d) => d.name));
  const ideas = [];
  if (names.has("sales")) ideas.push("Which region grew fastest in revenue from 2024 to 2025?", "Plot the monthly revenue trend");
  if (names.has("taxi_trips")) ideas.push("Do card payments get higher tips than cash? Test it.");
  if (names.has("employees")) ideas.push("What drives monthly salary? Fit a regression.");
  if (names.has("orders")) ideas.push("Top 5 customers by sales value in the shop database");
  if (!ideas.length && names.size) ideas.push("Give me an overview of this dataset", "Which columns have data quality problems?");
  $("#suggestions").innerHTML = ideas.slice(0, 4).map((q) => `<button type="button">${esc(q)}</button>`).join("");
}

// ------------------------------------------------------------------ agent card
function userBubble(text) {
  const d = document.createElement("div");
  d.className = "msg-user";
  d.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  return d;
}

function newAgentCard() {
  const el = document.createElement("div");
  el.className = "card";
  el.innerHTML = `<div class="card-body">
      <div class="status-line"><span class="spinner"></span><span class="status-text">Starting…</span></div>
      <div class="plan" hidden></div>
      <div class="steps"></div>
      <div class="notes critic" hidden></div>
      <div class="answer" hidden></div>
    </div>`;
  const body = $(".card-body", el);
  const steps = $(".steps", el);
  const running = new Map();
  const charts = [];

  const api_ = {
    el,
    status(text) { $(".status-text", el).textContent = text; },
    plan(p) {
      const box = $(".plan", el);
      box.hidden = false;
      box.innerHTML = `<div class="section-label">Plan <span class="chip accent">${esc(p.intent || "analysis")}</span></div>
        <ol>${(p.plan || []).map((s) => `<li>${esc(s.goal)}</li>`).join("")}</ol>`;
    },
    stepStart(ev) {
      const d = document.createElement("details");
      d.className = "step";
      d.innerHTML = `<summary><span class="step-icon run"><span class="spinner"></span></span>
        <span class="purpose">${esc(ev.purpose || ev.tool)}</span><span class="dur">running…</span></summary>
        <div class="step-body"><pre class="code">${highlight(ev.code || "", ev.tool === "run_sql" ? "sql" : "python")}</pre></div>`;
      steps.appendChild(d);
      running.set(ev.index, d);
    },
    step(s) {
      let d = running.get(s.index);
      if (!d) { d = document.createElement("details"); d.className = "step"; steps.appendChild(d); }
      running.delete(s.index);
      const lang = s.type === "sql" ? "sql" : "python";
      const label = s.type === "sql" ? `SQL · ${esc(s.target || "files")}` : s.type === "describe" ? "Profile" : "Python";
      const warn = (s.verifier || []).length;
      d.innerHTML = `<summary><span class="step-icon ${s.ok ? "ok" : "err"}">${s.ok ? icon.ok : icon.err}</span>
        <span class="purpose">${esc(s.purpose || label)}</span>
        <span class="chip">${label}</span>${warn ? `<span class="chip warn">${warn} check${warn > 1 ? "s" : ""}</span>` : ""}
        ${s.ok ? "" : `<span class="chip err">${esc(s.error_type || "error")}</span>`}
        <span class="dur">${s.duration_s != null ? s.duration_s.toFixed(2) + "s" : ""}</span></summary>
        <div class="step-body"></div>`;
      const b = $(".step-body", d);
      if (s.type !== "describe") b.insertAdjacentHTML("beforeend", `<pre class="code">${highlight(s.code || "", lang)}</pre>`);
      if (s.stdout) b.insertAdjacentHTML("beforeend", `<pre class="out">${esc(s.stdout)}</pre>`);
      if (s.value) b.insertAdjacentHTML("beforeend", `<pre class="out">${esc(s.value)}</pre>`);
      for (const t of s.tables || []) b.insertAdjacentHTML("beforeend", tableHTML(t));
      if (s.error) b.insertAdjacentHTML("beforeend", `<pre class="out err">${esc(s.error)}</pre>`);
      if (warn) b.insertAdjacentHTML("beforeend", `<div class="notes"><strong>Verifier</strong><ul>${s.verifier.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></div>`);
      if (s.ok) charts.push(...(s.charts || []));
    },
    critic(ev) {
      const box = $(".critic", el);
      box.hidden = false;
      box.className = "notes critic";
      box.innerHTML = ev.verdict === "revise"
        ? `<strong>Critic asked for a revision:</strong> ${esc((ev.issues || []).join("; "))}`
        : "";
      if (ev.verdict !== "revise") box.hidden = true;
    },
    verifier(msg) {
      const box = $(".critic", el);
      box.hidden = false;
      box.innerHTML = `<strong>Verifier:</strong> ${esc(msg)} — sent back to the coder.`;
    },
    clarify(q) {
      const a = $(".answer", el);
      a.hidden = false;
      a.innerHTML = `<div class="clarify"><div class="section-label" style="color:inherit">Clarification needed</div>${md(q)}</div>`;
    },
    answer(ev) {
      const a = $(".answer", el);
      a.hidden = false;
      a.classList.toggle("failed", ev.status === "failed");
      a.innerHTML = `<div class="section-label">Answer ${ev.status === "failed" ? '<span class="chip err">not completed</span>' : '<span class="chip ok">verified against executed code</span>'}</div>
        <div class="markdown">${md(ev.answer_md)}</div><div class="answer-charts"></div>`;
      const box = $(".answer-charts", a);
      for (const c of charts) renderChart(box, c);
    },
    error(msg) {
      const a = $(".answer", el);
      a.hidden = false;
      a.classList.add("failed");
      a.innerHTML = `<div class="section-label">Error</div><div class="markdown"><p>${esc(msg)}</p></div>`;
    },
    done(u) {
      $(".status-line", el)?.remove();
      for (const d of running.values()) d.remove();
      if (u && u.llm_calls !== undefined) {
        body.insertAdjacentHTML("afterend", `<div class="card-foot">
          <span>${u.steps ?? 0} code step${u.steps === 1 ? "" : "s"}${u.failed_steps ? ` (${u.failed_steps} retried)` : ""}</span>
          <span>${u.llm_calls} LLM calls</span><span>${((u.input_tokens + u.output_tokens) / 1000).toFixed(1)}k tokens</span>
          <span>$${(u.cost_usd || 0).toFixed(4)}</span><span>${(u.latency_s || 0).toFixed(1)} s</span>
          <span>${esc((u.model || "").replace("gemini/", ""))}</span></div>`);
      }
    },
  };
  return api_;
}

// ------------------------------------------------------------------ asking
async function ask(question) {
  if (state.busy || !question.trim()) return;
  const s = await ensureSession();
  state.busy = true;
  $("#sendBtn").disabled = true;
  $("#emptyState")?.remove();
  const thread = $("#thread");
  thread.appendChild(userBubble(question));
  const card = newAgentCard();
  thread.appendChild(card.el);
  thread.scrollTop = thread.scrollHeight;
  let usage = null;
  try {
    const res = await fetch(`/api/sessions/${s.id}/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, model: $("#modelSelect").value, critic: $("#criticToggle").checked }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const data = block.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5)).join("\n");
        if (!data) continue;
        const ev = JSON.parse(data);
        handleEvent(card, ev);
        if (ev.type === "usage") usage = ev.usage;
        const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 240;
        if (nearBottom) thread.scrollTop = thread.scrollHeight;
      }
    }
  } catch (e) {
    card.error(e.message);
  } finally {
    card.done(usage);
    state.busy = false;
    $("#sendBtn").disabled = false;
    state.session = await api(`/api/sessions/${s.id}`);
    $("#sessionTitle").textContent = state.session.title;
    renderUsage();
    refreshSessionList();
  }
}

function handleEvent(card, ev) {
  switch (ev.type) {
    case "status": card.status(ev.message); break;
    case "plan": card.plan(ev); card.status("Writing code"); break;
    case "step_start": card.stepStart(ev); card.status("Running code in the sandbox"); break;
    case "step": card.step(ev.step); card.status(ev.step.ok ? "Checking the result" : "Fixing an error"); break;
    case "critic": card.critic(ev); break;
    case "verifier": card.verifier(ev.message); break;
    case "clarification": card.clarify(ev.question); break;
    case "answer": card.answer(ev); break;
    case "error": card.error(ev.message); break;
    default: break;
  }
}

// ------------------------------------------------------------------ uploads & connections
async function uploadFiles(files) {
  const s = await ensureSession();
  const prog = $("#uploadProgress");
  for (const f of files) {
    prog.hidden = false;
    prog.textContent = `Uploading and profiling ${f.name}…`;
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await api(`/api/sessions/${s.id}/datasets`, { method: "POST", body: fd });
      toast(`Loaded ${r.dataset.name}: ${r.dataset.rows.toLocaleString()} rows`);
    } catch (e) {
      toast(`${f.name}: ${e.message}`, true);
    }
  }
  prog.hidden = true;
  state.session = await api(`/api/sessions/${s.id}`);
  renderDatasets();
  renderSuggestions();
}

// ------------------------------------------------------------------ data drawer
async function openDataset(name) {
  const s = state.session;
  const data = await api(`/api/sessions/${s.id}/datasets/${encodeURIComponent(name)}/profile`);
  state.drawer = { name, profile: data.profile, preview: null };
  $("#drawerTitle").textContent = name;
  $("#cleanCount").textContent = data.profile.cleaning_suggestions?.length || "";
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  $("#scrim").classList.add("show");
  showTab("columns");
}

function closeDrawer() {
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
  $("#scrim").classList.remove("show");
  $("#sidebar").classList.remove("open");
}

async function showTab(tab) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tab));
  const p = state.drawer.profile;
  const body = $("#drawerBody");
  if (tab === "columns") {
    const rows = p.column_profiles.map((c) => {
      let stats = "";
      if (c.kind === "numeric") stats = `min ${fmtNum(c.min)} · mean ${fmtNum(c.mean)} · max ${fmtNum(c.max)}`;
      else if (c.kind === "datetime") stats = `${String(c.min).slice(0, 10)} → ${String(c.max).slice(0, 10)}`;
      else stats = (c.top || []).slice(0, 3).map((t) => `${t.value} (${t.count})`).join(", ");
      const hist = c.histogram?.counts || [];
      const maxH = Math.max(1, ...hist);
      const spark = hist.length ? `<div class="spark">${hist.map((h) => `<i style="height:${Math.max(4, (h / maxH) * 100)}%"></i>`).join("")}</div>` : "";
      return `<div class="col-row"><div><div class="cname" title="${esc(c.name)}">${esc(c.name)}${c.is_candidate_key ? ' <span class="chip accent">key</span>' : ""}</div>
          <div class="ctype muted">${esc(c.dtype)} · ${c.unique.toLocaleString()} unique</div></div>
        <div title="${c.null_pct}% missing"><div class="nullbar"><i style="width:${c.null_pct}%"></i></div><div class="ctype muted">${c.null_pct}% null</div></div>
        <div class="cstats" title="${esc(stats)}">${esc(stats)}</div><div class="spark-cell">${spark}</div></div>`;
    });
    body.innerHTML = `<div class="stats">
        <div class="stat"><b>${p.rows.toLocaleString()}</b><span>rows</span></div>
        <div class="stat"><b>${p.columns}</b><span>columns</span></div>
        <div class="stat"><b>${p.duplicate_rows.toLocaleString()}</b><span>duplicate rows</span></div>
        <div class="stat"><b>${p.memory_mb}</b><span>MB in memory</span></div></div>${rows.join("")}`;
  } else if (tab === "cleaning") {
    const tips = p.cleaning_suggestions || [];
    body.innerHTML = tips.length ? tips.map((t, i) => `<div class="tip"><div class="tip-head">
        <span class="chip ${t.severity === "high" ? "err" : t.severity === "medium" ? "warn" : ""}">${esc(t.type.replace("_", " "))}</span>
        <span>${md(t.message).replace(/<\/?p>/g, "")}</span>
        <button class="btn small copy" data-ask="${i}">Ask agent to fix</button></div>
        <pre class="code">${highlight(t.code, "python")}</pre></div>`).join("")
      : '<p class="muted">No data-quality problems detected.</p>';
  } else {
    body.innerHTML = '<div class="status-line"><span class="spinner"></span>Loading preview…</div>';
    if (!state.drawer.preview) {
      state.drawer.preview = await api(`/api/sessions/${state.session.id}/datasets/${encodeURIComponent(state.drawer.name)}/preview?rows=100`);
    }
    body.innerHTML = tableHTML(state.drawer.preview, 100);
  }
}

// ------------------------------------------------------------------ wiring
function wire() {
  $("#newSession").onclick = () => newSession().catch((e) => toast(e.message, true));
  $("#sessionList").onclick = async (e) => {
    const del = e.target.closest("[data-del]");
    if (del) {
      e.stopPropagation();
      if (!confirm("Delete this analysis and its uploaded data?")) return;
      await api(`/api/sessions/${del.dataset.del}`, { method: "DELETE" });
      if (state.session?.id === del.dataset.del) { state.session = null; await newSession(); } else refreshSessionList();
      return;
    }
    const li = e.target.closest("li[data-id]");
    if (li) { openSession(li.dataset.id); $("#sidebar").classList.remove("open"); }
  };
  $("#fileInput").onchange = (e) => { uploadFiles([...e.target.files]); e.target.value = ""; };
  const dz = $("#dropzone");
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
  dz.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));
  $("#datasetList").onclick = (e) => { const li = e.target.closest("[data-ds]"); if (li) openDataset(li.dataset.ds); };
  $("#connectForm").onsubmit = async (e) => {
    e.preventDefault();
    const s = await ensureSession();
    try {
      const r = await api(`/api/sessions/${s.id}/connections`, { method: "POST", body: JSON.stringify({ kind: $("#dbKind").value, target: $("#dbTarget").value, name: $("#dbName").value || null }) });
      state.session = r.session;
      renderDatasets(); renderSuggestions();
      toast(`Connected ${r.connection.name}: ${r.connection.tables.length} tables (read-only)`);
    } catch (err) { toast(err.message, true); }
  };
  $("#dbKind").onchange = (e) => { $("#dbTarget").placeholder = e.target.value === "postgres" ? "postgresql://reader:pass@host:5432/db" : "path/to/file." + (e.target.value === "duckdb" ? "duckdb" : "sqlite"); };
  $("#loadSamples").onclick = async () => {
    const s = await ensureSession();
    const btn = $("#loadSamples");
    btn.disabled = true; btn.textContent = "Loading demo data…";
    try { state.session = await api(`/api/sessions/${s.id}/samples`, { method: "POST" }); renderDatasets(); renderSuggestions(); toast("Demo datasets loaded"); }
    catch (e) { toast(e.message, true); }
    finally { btn.disabled = false; btn.textContent = "Load demo datasets"; }
  };
  const ta = $("#question");
  ta.addEventListener("input", () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 180) + "px"; });
  ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#askForm").requestSubmit(); } });
  $("#askForm").onsubmit = (e) => { e.preventDefault(); const q = ta.value; ta.value = ""; ta.style.height = "auto"; ask(q); };
  $("#suggestions").onclick = (e) => { const b = e.target.closest("button"); if (b) ask(b.textContent); };
  $("#modelSelect").onchange = (e) => localStorageSet("model", e.target.value);
  $("#criticToggle").onchange = (e) => localStorageSet("critic", e.target.checked ? "1" : "0");
  $("#exportBtn").onclick = (e) => { e.stopPropagation(); $("#exportMenu").hidden = !$("#exportMenu").hidden; };
  document.addEventListener("click", () => ($("#exportMenu").hidden = true));
  $("#exportMenu").onclick = (e) => {
    const b = e.target.closest("[data-fmt]");
    if (!b || !state.session) return;
    window.location.href = `/api/sessions/${state.session.id}/export?format=${b.dataset.fmt}`;
  };
  $("#drawerClose").onclick = closeDrawer;
  $("#scrim").onclick = closeDrawer;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => showTab(t.dataset.tab)));
  $("#drawerBody").onclick = (e) => {
    const b = e.target.closest("[data-ask]");
    if (!b) return;
    const tip = state.drawer.profile.cleaning_suggestions[+b.dataset.ask];
    closeDrawer();
    ask(`In table ${state.drawer.name}: ${tip.message.replace(/`/g, "")} Fix this issue, save the cleaned table as ${state.drawer.name}_clean, and show how many rows changed.`);
  };
  $("#menuBtn").onclick = () => { $("#sidebar").classList.add("open"); $("#scrim").classList.add("show"); };
}

(async function init() {
  wire();
  try {
    await loadConfig();
    const last = localStorageGet("session");
    if (last) await openSession(last).catch(() => refreshSessionList());
    else await refreshSessionList();  // a session is created on the first upload or question
  } catch (e) {
    toast("Cannot reach the backend: " + e.message, true);
  }
})();
