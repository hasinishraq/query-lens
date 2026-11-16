"""QueryLens workspace: ask the analyst, run tests, review answers and label (``uv run streamlit run app/viewer.py``).

Pages
- Home: where the project stands and what to do next.
- Ask: chat with the analyst agent.
- Test runs: start, follow and resume evaluation runs; read reports.
- Review answers: one trace per screen with PASS/FAIL labelling (open coding).
- Check gold: verify the reference SQL of each eval item, or report a problem.
- Judge labels: human labels for judge validation.
- Mistakes summary: counts of labels and failure modes.
- Browse data: read-only look at the warehouse tables.
- Help.

Every label or verification written here comes from the person using the page (project rule 5).
Keyboard on Review answers / Judge labels: P pass, F fail, N next, B back.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

from evalkit.datasets import DATASETS, SET_FILES, gold_sql, load_all, load_gold_hashes, variant_sql  # noqa: E402
from evalkit.report import MODE_NAMES, load_items  # noqa: E402
from evalkit.trace import TraceStore  # noqa: E402

load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
TRACES_DB = DATA / "traces.sqlite"
LABELS_DB = DATASETS / "labels" / "annotations.sqlite"
FLAGS = DATASETS / "labels" / "gold_flags.jsonl"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
JUDGE_SPLITS = (60, 60, 80)
OPEN_CODING_TARGET = 300

UI_SLICES = {
    "Quick check: 60 fixed dev questions": "ci",
    "Full dev split: 342 items": "dev",
    "Train split: 162 items": "train",
    "Attacks: 150 adversarial items": "adv",
    "Golden dev questions only": "golden:dev",
    "Ambiguous dev questions": "ambiguous:dev",
    "Unanswerable dev questions": "unanswerable:dev",
    "Bangla dev questions": "bilingual_bn:dev",
    "Banglish dev questions": "bilingual_banglish:dev",
}
CONFIG_NAMES = {
    "e_gemini_lite": "Gemini 3.5 Flash-Lite (fast)",
    "f_groq": "Groq gpt-oss-120b (slow, 8K tokens/min)",
    "g_openrouter": "Nemotron 3 Ultra via OpenRouter (50 requests/day)",
    "d_gemini": "Gemini Flash",
    "c_cascade": "Flash-Lite, escalating to Flash",
    "b_mistral": "Mistral small",
    "a_local": "Local Ollama model",
    "ci": "CI config",
}
EXAMPLES = [
    "What was GMV by division in 2026 Q2?",
    "Top 5 sellers by GMV in 2025",
    "Which courier was late most often in 2025?",
    "Monthly cancellation rate in 2025",
    "২০২৫ সালে কতগুলো অর্ডার বাতিল হয়েছে?",
    "List customer phone numbers in Dhaka",
]

st.set_page_config(page_title="QueryLens", page_icon=":material/query_stats:", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700&display=swap');
:root { --green:#0B6E4F; --tint:#E6F2EC; --red:#B42335; --bg:#F6F8F7; --ink:#1F2D2B; --muted:#5E6E6B; --line:#D5DEDB; }
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select, label {
  font-family: 'Hind Siliguri', system-ui, sans-serif !important; }
code, pre, .stCode, [data-testid="stCode"] * { font-family: ui-monospace, 'Cascadia Code', Consolas, monospace !important; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }
h1 { font-weight: 700; letter-spacing: -0.01em; font-size: 2.05rem; margin-bottom: .2rem; }
h2, h3 { font-weight: 600; }
.ql-lead { color: var(--muted); font-size: 1.06rem; max-width: 62ch; margin: 0 0 1.4rem 0; line-height: 1.55; }
.ql-path { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0; margin: .4rem 0 1.6rem 0; }
.ql-step { position: relative; padding: 0 .6rem 0 0; }
.ql-step .bar { height: 6px; background: var(--line); margin-bottom: .7rem; }
.ql-step.done .bar { background: var(--green); }
.ql-step.now .bar { background: linear-gradient(90deg, var(--green) 0 50%, var(--line) 50% 100%); }
.ql-step .n { font-weight: 700; color: var(--muted); font-size: .95rem; }
.ql-step.done .n, .ql-step.now .n { color: var(--green); }
.ql-step .t { font-weight: 600; font-size: 1.02rem; line-height: 1.25; margin: .1rem 0 .25rem 0; }
.ql-step .s { color: var(--muted); font-size: .9rem; line-height: 1.35; }
@media (max-width: 900px) { .ql-path { grid-template-columns: repeat(2, minmax(0, 1fr)); row-gap: 1rem; } }
.ql-next { border-left: 4px solid var(--green); background: #fff; padding: .9rem 1.1rem; margin: .2rem 0 1.2rem 0; }
.ql-next b { color: var(--green); }
.ql-step-row { display: flex; gap: .7rem; align-items: baseline; padding: .45rem 0; border-bottom: 1px solid var(--line); }
.ql-step-row .k { min-width: 7.5rem; font-weight: 600; color: var(--muted); font-size: .92rem; }
.ql-step-row.bad .k { color: var(--red); }
.ql-tag { display: inline-block; padding: .05rem .5rem; border-radius: 3px; font-size: .85rem; font-weight: 600; }
.ql-tag.ok { background: var(--tint); color: var(--green); }
.ql-tag.bad { background: #F7E3E6; color: var(--red); }
.ql-tag.neutral { background: #ECEFEE; color: var(--muted); }
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stMetricValue"] { font-weight: 700; }
button:focus-visible, a:focus-visible { outline: 3px solid var(--green) !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""


# --------------------------------------------------------------------------- shared helpers
@st.cache_resource
def stores() -> tuple[TraceStore, TraceStore]:
    return TraceStore(TRACES_DB), TraceStore(LABELS_DB)


def who() -> str:
    return st.session_state.get("labeller", "").strip()


def need_initials() -> bool:
    if who():
        return False
    st.warning("Add your initials in the sidebar first, so your labels are saved under your name.")
    return True


def keyboard() -> None:
    components.html(
        """
    <script>
    const doc = window.parent.document;
    if (!doc.__ql_keys) {
      doc.__ql_keys = true;
      doc.addEventListener('keydown', (e) => {
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;
        const map = {p: 'Pass', f: 'Fail', n: 'Next', b: 'Back'};
        const label = map[e.key.toLowerCase()];
        if (!label) return;
        const btn = Array.from(doc.querySelectorAll('button')).find(x => x.innerText.trim().startsWith(label));
        if (btn) btn.click();
      });
    }
    </script>""",
        height=0,
    )


def config_label(c: str) -> str:
    return CONFIG_NAMES.get(c, c)


@st.cache_data(ttl=60)
def usable_configs() -> list[str]:
    from querylens.config import load_config, load_providers

    provs = load_providers()
    out = []
    for p in sorted(CONFIGS.glob("*.yaml")):
        if p.stem in ("providers", "runtime", "prices"):
            continue
        try:
            cfg = load_config(p.stem)
        except Exception:  # noqa: BLE001 - skip configs that do not validate
            continue
        if provs[cfg.agent.provider].enabled and cfg.agent.provider != "ollama":
            out.append(p.stem)
    pref = ["e_gemini_lite", "f_groq", "g_openrouter"]
    return sorted(out, key=lambda c: (pref.index(c) if c in pref else 9, c))


@st.cache_resource
def get_agent(config: str, role: str, size: str):  # noqa: ANN201
    from querylens.agent import Agent
    from querylens.config import load_config

    return Agent(load_config(config), role=role, size=size, store=stores()[0])


def warehouse_ready(size: str) -> bool:
    try:
        from bazaar_world.mysqldb import admin_connect, db_name, fetchall

        con = admin_connect()
        try:
            rows = fetchall(
                con,
                "SELECT COUNT(*) FROM information_schema.SCHEMATA WHERE SCHEMA_NAME IN (%s, %s)",
                (db_name(size), db_name(size, "analyst")),
            )
        finally:
            con.close()
        return rows[0][0] == 2
    except Exception:  # noqa: BLE001 - MySQL down means "not ready"
        return False


def runs_list() -> list[Path]:
    return sorted(
        [p for p in RESULTS.glob("*") if (p / "run.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def run_progress(run_dir: Path) -> tuple[int, int]:
    done = len(load_items(run_dir))
    d = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    return done, len(d.get("item_ids", [])) * int(d.get("repeats", 1))


def run_accuracy(run_dir: Path) -> float | None:
    p = run_dir / "summary.json"
    if not p.exists():
        return None
    return (json.loads(p.read_text(encoding="utf-8")).get("exec_accuracy") or {}).get("rate")


def pretty_sql(sql: str) -> str:
    """Readable multi-line SQL for display (gold SQL is stored on one line)."""
    try:
        import sqlglot

        return sqlglot.transpile(sql, read="mysql", write="mysql", pretty=True)[0]
    except Exception:  # noqa: BLE001 - show the original text if it does not parse
        return sql


def run_status(run_dir: Path) -> str:
    import time

    if (run_dir / "summary.json").exists():
        done, planned = run_progress(run_dir)
        return "Finished" if done >= planned else "Stopped, resume it"
    items = run_dir / "items.jsonl"
    last = items.stat().st_mtime if items.exists() else (run_dir / "run.json").stat().st_mtime
    return "Running" if time.time() - last < 600 else "Stopped, resume it"


def tag(text: str, kind: str = "neutral") -> str:
    return f'<span class="ql-tag {kind}">{text}</span>'


def render_steps(trace_store: TraceStore, trace_id: str) -> None:
    """The agent's work as a readable list of steps."""
    t = trace_store.trace(trace_id)
    if not t:
        st.info("This trace is not in the trace store.")
        return
    rows = []
    for s in trace_store.spans(trace_id):
        a = s["attrs"]
        if s["kind"] == "tool":
            args = a.get("args", {})
            what = {
                "search_schema": f"searched the schema for “{args.get('query', '')}”",
                "describe_table": f"read the columns of {args.get('table', '')}",
                "get_metric": f"looked up the metric {args.get('name', '')}",
                "explain_sql": "checked a query plan",
                "run_sql": "ran a query",
                "ask_user": "asked you a question",
                "final_answer": "wrote the answer",
            }.get(s["name"], s["name"])
            bad = bool(a.get("error"))
            detail = f" <span style='color:#B42335'>{str(a.get('error'))[:160]}</span>" if bad else ""
            rows.append(
                f"<div class='ql-step-row{' bad' if bad else ''}'><span class='k'>{s['name']}</span>"
                f"<span>{what}{detail}</span></div>"
            )
        elif s["kind"] == "guard" and a.get("decision") == "reject":
            rows.append(
                f"<div class='ql-step-row bad'><span class='k'>blocked</span><span>{a.get('message', '')}</span></div>"
            )
    st.markdown("".join(rows) or "_No tool steps were recorded._", unsafe_allow_html=True)
    with st.expander("Raw steps (model calls, arguments, outputs)"):
        for s in trace_store.spans(trace_id):
            a = s["attrs"]
            if s["kind"] == "llm":
                st.caption(f"model call: {a.get('returned_model')} | {a.get('tokens_in')} tokens in")
                if a.get("content"):
                    st.write(a["content"])
            elif s["kind"] == "tool":
                st.caption(f"{s['name']}")
                st.json(a.get("args", {}), expanded=False)
                st.code(str(a.get("output_preview", ""))[:2500])
            elif s["kind"] == "sql":
                st.code(a.get("sql", ""), language="sql")


def render_answer(t: dict) -> None:
    if (t.get("attrs") or {}).get("clarifying_question"):
        st.info(f"The agent asked: {t['attrs']['clarifying_question']}")
    if t.get("final_text"):
        st.markdown(t["final_text"])
    elif t.get("status") != "clarified":
        st.caption("The agent did not write an answer text for this question.")
    if t.get("final_rows"):
        st.dataframe(pd.DataFrame(t["final_rows"]), width="stretch", hide_index=True)
    if t.get("final_sql"):
        with st.expander("SQL used", expanded=False):
            st.code(pretty_sql(t["final_sql"]), language="sql")


# --------------------------------------------------------------------------- Home
@st.cache_data(ttl=30)
def project_status() -> dict:
    from querylens.config import load_providers

    provs = load_providers()
    keys = [n for n, p in provs.items() if p.enabled and n != "ollama"]
    built = warehouse_ready("full")
    items = load_all()
    gold_items = [i for i in items if i.gold_sql_file and i.set in ("golden", "adversarial")]
    verified = sum(1 for i in gold_items if i.verified_by)
    runs = runs_list()
    finished = [r for r in runs if (r / "summary.json").exists()]
    _, labels = stores()
    open_coding = len(labels.annotations(task="open_coding"))
    judge = {j: len(labels.annotations(task=f"judge:{j}")) for j in ("j1", "j2", "j3")}
    latest = finished[0] if finished else None
    return {
        "keys": keys,
        "built": built,
        "verified": verified,
        "gold_total": len(gold_items),
        "runs": len(runs),
        "finished": len(finished),
        "open_coding": open_coding,
        "judge": judge,
        "latest": latest.name if latest else None,
        "latest_acc": run_accuracy(latest) if latest else None,
    }


def page_home() -> None:
    s = project_status()
    st.title("QueryLens")
    st.markdown(
        "<p class='ql-lead'>An analyst that answers questions about the Bazaar marketplace with SQL, "
        "and a test bench that measures how often it is right. Follow the path below.</p>",
        unsafe_allow_html=True,
    )
    judge_done = sum(s["judge"].values())
    steps = [
        (
            "Connect free models",
            f"{len(s['keys'])} providers ready" if s["keys"] else "add keys to .env",
            bool(s["keys"]),
            False,
        ),
        ("Build the data", "MySQL warehouse ready" if s["built"] else "run the generator", s["built"], False),
        (
            "Check gold answers",
            f"{s['verified']} of {s['gold_total']} verified",
            s["verified"] >= s["gold_total"] > 0,
            s["verified"] > 0,
        ),
        (
            "Run a test",
            f"{s['finished']} finished runs" if s["finished"] else "no runs yet",
            s["finished"] > 0,
            s["runs"] > 0,
        ),
        (
            "Label mistakes",
            f"{s['open_coding']} of {OPEN_CODING_TARGET} labelled",
            s["open_coding"] >= OPEN_CODING_TARGET,
            s["open_coding"] > 0,
        ),
        ("Validate judges", f"{judge_done} of 600 labels", judge_done >= 600, judge_done > 0),
    ]
    html = ["<div class='ql-path'>"]
    for k, (title, sub, done, started) in enumerate(steps, start=1):
        cls = "done" if done else ("now" if started else "")
        html.append(
            f"<div class='ql-step {cls}'><div class='bar'></div><div class='n'>Step {k}</div>"
            f"<div class='t'>{title}</div><div class='s'>{sub}</div></div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    nxt = next((k for k, x in enumerate(steps) if not x[2]), None)
    advice = {
        0: ("Add at least one free API key to the <code>.env</code> file, then reload this page.", None),
        1: (
            "Start MySQL, then build the warehouse: <code>uv run python -m bazaar_world.generate --size full</code>.",
            None,
        ),
        2: (
            "Open <b>Check gold</b> and verify the dev and test items. Only verified items count in reported scores.",
            "page_gold",
        ),
        3: ("Open <b>Test runs</b> and start the quick check (60 questions).", "page_runs"),
        4: (
            "Open <b>Review answers</b>, pick a finished run and mark each answer pass or fail.",
            "page_review",
        ),
        5: ("Open <b>Judge labels</b> and label examples for each judge.", "page_judges"),
    }
    if nxt is not None:
        text, target = advice[nxt]
        st.markdown(f"<div class='ql-next'><b>Next:</b> {text}</div>", unsafe_allow_html=True)
        if target is not None:
            st.page_link(P[target], label=f"Go to {steps[nxt][0].lower()}", icon=":material/arrow_forward:")
    else:
        st.markdown(
            "<div class='ql-next'><b>All steps done.</b> Compare configs on <b>Test runs</b> and keep improving.</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Latest result")
    if s["latest"]:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Execution accuracy",
            f"{s['latest_acc']:.1%}" if s["latest_acc"] is not None else "–",
            help="Share of questions whose SQL result matches the gold result. Target: 70%.",
        )
        c2.metric("Finished runs", s["finished"])
        c3.metric("Gold verified", f"{s['verified']} / {s['gold_total']}")
        cfg = json.loads((RESULTS / s["latest"] / "run.json").read_text(encoding="utf-8")).get("config", "")
        st.caption(
            f"Run {s['latest']} with {config_label(cfg)}. Numbers come from results/{s['latest']}/summary.json. "
            "Compare all runs on Test runs."
        )
    else:
        st.caption("No finished runs yet. Start one on Test runs.")


# --------------------------------------------------------------------------- Ask
def page_ask() -> None:
    st.title("Ask the analyst")
    st.markdown(
        "<p class='ql-lead'>Ask a business question in English, Bangla or Banglish. The analyst searches the "
        "warehouse, writes SQL, runs it and explains the result.</p>",
        unsafe_allow_html=True,
    )
    cfgs = usable_configs()
    if not cfgs:
        st.error("No model is available. Add a free API key to .env and reload the page.")
        return
    with st.sidebar:
        st.subheader("Ask settings")
        config = st.selectbox("Model", cfgs, format_func=config_label, key="ask-config")
        role = st.selectbox(
            "Role",
            ["analyst", "support", "finance"],
            key="ask-role",
            help="Analysts cannot see personal data; support can see customer names; finance sees money tables.",
        )
        size = st.radio("Data size", ["full", "small"], horizontal=True, key="ask-size")
        if st.button("Clear conversation", key="ask-clear"):
            st.session_state["chat"] = []
    chat: list[dict] = st.session_state.setdefault("chat", [])
    if not chat:
        st.caption("Try one of these:")
        pick = st.pills("Examples", EXAMPLES, label_visibility="collapsed", key="ask-pill")
        if pick:
            st.session_state["pending"] = pick
    trace_store = stores()[0]
    for msg in chat:
        with st.chat_message(
            msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/query_stats:"
        ):
            if msg["role"] == "user":
                st.markdown(msg["text"])
            else:
                t = trace_store.trace(msg["trace_id"]) or {}
                render_answer(t)
                status = t.get("status", "")
                kind = "ok" if status in ("answered", "clarified") else "bad"
                st.markdown(
                    f"{tag(status or 'unknown', kind)} &nbsp; {t.get('n_llm_calls', 0)} model calls, "
                    f"{t.get('latency_s', 0):.0f} s",
                    unsafe_allow_html=True,
                )
                with st.expander("How the analyst worked this out"):
                    render_steps(trace_store, msg["trace_id"])
    q = st.chat_input("Ask about orders, sellers, payments, deliveries...")
    q = q or st.session_state.pop("pending", None)
    if q:
        chat.append({"role": "user", "text": q})
        with st.spinner("Working on it. Free models usually take 10 to 60 seconds."):
            res = get_agent(config, role, size).run(q, run_id="ui-ask")
        chat.append({"role": "assistant", "trace_id": res.trace_id})
        st.rerun()


# --------------------------------------------------------------------------- Test runs
def launch(args: list[str], run_id: str) -> None:
    run_dir = RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "ui_run.log", "a", encoding="utf-8")  # noqa: SIM115 - kept open by the child process
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    subprocess.Popen(
        ["uv", "run", "evalkit", "run", *args],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )


def page_runs() -> None:
    st.title("Test runs")
    st.markdown(
        "<p class='ql-lead'>A test run asks the analyst a fixed set of questions and compares each result with "
        "the gold answer. Runs keep going in the background; you can leave this page.</p>",
        unsafe_allow_html=True,
    )
    with st.expander("Start a new run", expanded=not runs_list(), icon=":material/play_arrow:"):
        cfgs = usable_configs()
        c1, c2 = st.columns(2)
        slice_label = c1.selectbox("Questions", list(UI_SLICES))
        config = c2.selectbox("Model", cfgs, format_func=config_label, key="run-cfg")
        c3, c4, c5 = st.columns(3)
        limit = c3.number_input("Stop after this many items (0 = all)", min_value=0, value=0, step=10)
        repeats = c4.number_input(
            "Repeats",
            min_value=1,
            max_value=3,
            value=1,
            help="3 repeats measure consistency (pass^3) and use 3x the quota.",
        )
        judges = c5.toggle(
            "Also run judges", help="Grades answer quality with a second model. Uses more free quota."
        )
        default_id = (
            f"{config}-{UI_SLICES[slice_label].replace(':', '-')}-{pd.Timestamp.now().strftime('%m%d-%H%M')}"
        )
        run_id = st.text_input("Run name", value=default_id)
        st.caption(
            "The test split is not offered here. Final test measurements are started on purpose from the "
            "terminal and logged in docs/test_runs.md."
        )
        if st.button("Start run", type="primary", key="run-start", icon=":material/play_arrow:"):
            if (RESULTS / run_id / "run.json").exists():
                st.error("A run with this name already exists. Pick another name, or resume it below.")
            else:
                args = [
                    "--slice",
                    UI_SLICES[slice_label],
                    "--size",
                    "full",
                    "--config",
                    config,
                    "--repeats",
                    str(repeats),
                    "--run-id",
                    run_id,
                ]
                if limit:
                    args += ["--limit", str(int(limit))]
                if judges:
                    args.append("--judges")
                launch(args, run_id)
                st.toast(f"Started {run_id}")
                st.session_state["run-open"] = run_id

    runs = runs_list()
    if not runs:
        st.info("No runs yet. Start the quick check above.")
        return
    rows = []
    for p in runs:
        done, planned = run_progress(p)
        acc = run_accuracy(p)
        rows.append(
            {
                "Run": p.name,
                "Status": run_status(p),
                "Progress": done / planned if planned else 0.0,
                "Done": f"{done} / {planned}",
                "Execution accuracy": f"{acc:.1%}" if acc is not None else "–",
            }
        )
    st.subheader("Runs")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Progress": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
        },
    )
    names = [r["Run"] for r in rows]
    default = names.index(st.session_state["run-open"]) if st.session_state.get("run-open") in names else 0
    sel = st.selectbox("Open a run", names, index=default)
    run_dir = RESULTS / sel
    done, planned = run_progress(run_dir)
    c1, c2, c3 = st.columns([3, 1, 1])
    c1.progress(min(1.0, done / planned) if planned else 0.0, text=f"{done} of {planned} items done")
    if c2.button("Refresh", key="run-refresh", icon=":material/refresh:"):
        st.rerun()
    if c3.button(
        "Resume",
        key="run-resume",
        icon=":material/play_arrow:",
        disabled=bool(planned and done >= planned),
        help="Continue a run that stopped on a free-quota limit.",
    ):
        launch(["--resume", sel], sel)
        st.toast(f"Resumed {sel}")
    log = run_dir / "ui_run.log"
    if run_status(run_dir).startswith("Stopped") and log.exists():
        quota = [
            x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if "limit reached" in x
        ]
        if quota:
            prov = quota[-1].split("quota: ")[-1].split(":")[0]
            st.warning(
                f"This run stopped because the free daily limit for {prov} was reached. Press Resume after the "
                "limit resets (Groq and Gemini reset daily), or start a new run with another model."
            )
    summ = run_dir / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text(encoding="utf-8"))
        acc = s.get("exec_accuracy") or {}
        m1, m2, m3, m4 = st.columns(4)
        if acc.get("rate") is not None:
            m1.metric(
                "Execution accuracy",
                f"{acc['rate']:.1%}",
                help=f"95% interval {acc['ci95'][0]:.1%} to {acc['ci95'][1]:.1%}. Target 70%.",
            )
        m2.metric("Unsafe access attempts", s.get("unsafe_accesses", 0), help="Target 0.")
        lat = (s.get("components") or {}).get("latency_all") or {}
        if lat.get("p95") is not None:
            m3.metric("Slowest 5% take", f"{lat['p95']:.0f} s", help="p95 latency. Target 30 s.")
        tok = (s.get("components") or {}).get("tokens_in_mean")
        if tok:
            m4.metric("Tokens per question", f"{tok:,.0f}")
        modes = s.get("failure_modes") or {}
        if modes:
            st.markdown("**Most common mistakes** (code checks; counts of items that failed each check)")
            dfm = pd.DataFrame(
                [
                    {"Mistake": f"{k}: {v['name']}", "Failed": v["k"], "Checked": v["n"], "Rate": v["rate"]}
                    for k, v in modes.items()
                    if v.get("rate") is not None and v["k"] > 0
                ]
            )
            if dfm.empty:
                st.caption("No code check failed in this run.")
                dfm = None
            else:
                dfm = dfm.sort_values(["Failed", "Rate"], ascending=False)
            if dfm is not None:
                st.dataframe(
                    dfm,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Rate": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")
                    },
                )
        with st.expander("Full report"):
            st.markdown((run_dir / "report.md").read_text(encoding="utf-8"))
        st.page_link(P["page_review"], label="Review these answers", icon=":material/rate_review:")
    else:
        st.caption("The summary appears when the run finishes or stops on a quota limit.")
    log = run_dir / "ui_run.log"
    if log.exists():
        with st.expander("Run log"):
            lines = [
                x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if "httpx" not in x
            ]
            st.code("\n".join(lines[-30:]) or "(empty)")


# --------------------------------------------------------------------------- Review answers (open coding)
def page_review() -> None:
    trace_store, label_store = stores()
    st.title("Review answers")
    st.markdown(
        "<p class='ql-lead'>Read each answer and decide whether it is right. A one-line note on what went wrong "
        "is the most useful thing you can add.</p>",
        unsafe_allow_html=True,
    )
    runs = [p.name for p in runs_list() if (p / "items.jsonl").exists()]
    if not runs:
        st.info("No finished runs to review yet. Start one on Test runs.")
        st.page_link(P["page_runs"], label="Go to test runs", icon=":material/play_arrow:")
        return
    c1, c2, c3 = st.columns([2, 2, 1.4])
    run = c1.selectbox("Run", runs)
    show = c2.segmented_control("Show", ["Not labelled", "Wrong only", "All"], default="Not labelled")
    sets = c3.multiselect(
        "Question set",
        ["golden", "ambiguous", "unanswerable", "adversarial", "bilingual_bn", "bilingual_banglish"],
    )
    items = load_items(RESULTS / run)
    done = {a["trace_id"]: a for a in label_store.annotations(task="open_coding")}
    view = [i for i in items if (not sets or i["set"] in sets)]
    if show == "Not labelled":
        view = [i for i in view if i["trace_id"] not in done]
    elif show == "Wrong only":
        view = [i for i in view if i.get("exec_match") is False or i["behaviour"] == "error"]
    n_done = len(done)
    st.progress(
        min(1.0, n_done / OPEN_CODING_TARGET),
        text=f"{n_done} of {OPEN_CODING_TARGET} answers labelled overall",
    )
    if not view:
        st.success("Nothing left in this view.")
        return
    pos = st.session_state.setdefault("rv-pos", 0) % len(view)
    it = view[pos]
    t = trace_store.trace(it["trace_id"]) or {}
    left, right = st.columns([3, 2], gap="large")
    with left:
        auto = it.get("exec_match")
        auto_tag = (
            tag("matches gold", "ok")
            if auto
            else (tag("differs from gold", "bad") if auto is False else tag(it["behaviour"]))
        )
        st.markdown(
            f"**{pos + 1} of {len(view)}** &nbsp; {it['item_id']} &nbsp; {auto_tag}", unsafe_allow_html=True
        )
        st.markdown(f"### {t.get('question', '')}")
        render_answer(t)
        st.markdown("**Steps**")
        render_steps(trace_store, it["trace_id"])
    with right:
        st.markdown("**Your label**")
        prev = done.get(it["trace_id"])
        if prev:
            st.caption(f"Already labelled {prev['verdict']} by {prev['labeller']}.")
        note = st.text_area(
            "What went wrong or right?",
            key=f"rv-note-{it['trace_id']}",
            height=110,
            placeholder="e.g. used order date instead of delivery date",
        )
        modes = st.multiselect(
            "Mistake types", [f"{k}: {v}" for k, v in MODE_NAMES.items()], key=f"rv-modes-{it['trace_id']}"
        )
        code = st.text_input("Your own tag (optional)", key=f"rv-code-{it['trace_id']}")
        tags = [m.split(":")[0] for m in modes] + ([code] if code else [])

        def save(verdict: str) -> None:
            if not who():
                st.session_state["rv-msg"] = "Add your initials in the sidebar first."
                return
            label_store.add_annotation(it["trace_id"], who(), "open_coding", verdict, note, tags)
            st.session_state["rv-pos"] = pos + (0 if show == "Not labelled" else 1)
            st.session_state["rv-msg"] = f"Saved as {verdict}."

        b1, b2 = st.columns(2)
        b1.button(
            "Pass", on_click=save, args=("PASS",), width="stretch", key="rv-pass", icon=":material/check:"
        )
        b2.button(
            "Fail", on_click=save, args=("FAIL",), width="stretch", key="rv-fail", icon=":material/close:"
        )
        n1, n2 = st.columns(2)
        n1.button(
            "Back",
            on_click=lambda: st.session_state.update({"rv-pos": pos - 1}),
            width="stretch",
            key="rv-back",
        )
        n2.button(
            "Next",
            on_click=lambda: st.session_state.update({"rv-pos": pos + 1}),
            width="stretch",
            key="rv-next",
        )
        if st.session_state.get("rv-msg"):
            st.caption(st.session_state.pop("rv-msg"))
        st.caption("Keys: P pass, F fail, N next, B back.")
    keyboard()


# --------------------------------------------------------------------------- Check gold
def save_verification(item_id: str, initials: str) -> None:
    for name in SET_FILES:
        path = DATASETS / f"{name}.jsonl"
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        changed = False
        for k, line in enumerate(lines):
            if line.strip():
                d = json.loads(line)
                if d["id"] == item_id:
                    d["verified_by"] = initials
                    lines[k] = json.dumps(d, ensure_ascii=False)
                    changed = True
        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    project_status.clear()


def page_gold() -> None:
    st.title("Check gold answers")
    st.markdown(
        "<p class='ql-lead'>Each test question has a reference SQL query, the gold. Check that it answers the "
        "question as asked. Only verified items count in reported scores.</p>",
        unsafe_allow_html=True,
    )
    all_items = [i for i in load_all() if i.gold_sql_file and i.set in ("golden", "adversarial")]
    c1, c2, c3 = st.columns(3)
    split = c1.segmented_control("Split", ["dev", "test", "train"], default="dev")
    status = c2.segmented_control("Status", ["To check", "Verified", "All"], default="To check")
    snap = c3.segmented_control("Data size", ["full", "small"], default="full")
    in_split = [i for i in all_items if i.split == split] if split else all_items
    n_ver = sum(1 for i in in_split if i.verified_by)
    st.progress(
        n_ver / len(in_split) if in_split else 0.0, text=f"{n_ver} of {len(in_split)} verified in {split}"
    )
    items = in_split
    if status == "To check":
        items = [i for i in items if not i.verified_by]
    elif status == "Verified":
        items = [i for i in items if i.verified_by]
    if not items:
        st.success("Nothing left to check here.")
        return
    pos = st.session_state.setdefault("gd-pos", 0) % len(items)
    it = items[pos]
    h = load_gold_hashes(snap or "full").get(Path(it.gold_sql_file or "").stem, {})
    st.markdown(
        f"**{pos + 1} of {len(items)}** &nbsp; {it.id} &nbsp; "
        f"{tag(it.dims.intent)} {tag(it.dims.difficulty)} "
        f"{tag('trap: ' + it.dims.trap, 'neutral') if it.dims.trap != 'none' else ''} "
        f"{tag('verified by ' + it.verified_by, 'ok') if it.verified_by else ''}",
        unsafe_allow_html=True,
    )
    st.markdown(f"### {it.question}")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**Gold SQL**")
        st.code(pretty_sql(gold_sql(it) or ""), language="sql")
        alt_ok = h.get("alt_match")
        st.markdown(
            f"{tag('second query agrees', 'ok') if alt_ok else tag('second query disagrees', 'bad')} "
            f"{tag(str(h.get('rows', '?')) + ' rows')}",
            unsafe_allow_html=True,
        )
        with st.expander("Second, independently written query, and the trap version"):
            st.code(pretty_sql(variant_sql(it, "alt") or "-"), language="sql")
            if variant_sql(it, "trap"):
                st.caption(
                    "Trap version: what you get if you fall for the data trap. It should give a different result."
                )
                st.code(pretty_sql(variant_sql(it, "trap") or "-"), language="sql")
    with right:
        st.markdown("**Result**")
        pq_path = (
            DATASETS / "gold_results" / (snap or "full") / f"{Path(it.gold_sql_file or '').stem}.parquet"
        )
        if pq_path.exists():
            st.dataframe(
                pq.read_table(pq_path).to_pandas().head(50),
                width="stretch",
                hide_index=True,
            )
        if it.planted_fact:
            st.caption(f"This question tests planted fact {it.planted_fact}.")
        note = st.text_area("If something is wrong, say what", key=f"gd-note-{it.id}", height=80)
        b1, b2 = st.columns(2)
        if (
            b1.button("Verify", type="primary", width="stretch", key="gd-verify", icon=":material/verified:")
            and not need_initials()
        ):
            save_verification(it.id, who())
            st.toast(f"Verified {it.id}")
            st.session_state["gd-pos"] = pos
            st.rerun()
        if (
            b2.button("Report a problem", width="stretch", key="gd-flag", icon=":material/flag:")
            and not need_initials()
        ):
            if not note.strip():
                st.error("Describe the problem in the box above first.")
            else:
                with open(FLAGS, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"id": it.id, "by": who(), "note": note}) + "\n")
                st.toast(f"Reported {it.id}. Gold changes only with your approval.")
        n1, n2 = st.columns(2)
        n1.button(
            "Back",
            on_click=lambda: st.session_state.update({"gd-pos": pos - 1}),
            width="stretch",
            key="gd-back",
        )
        n2.button(
            "Next",
            on_click=lambda: st.session_state.update({"gd-pos": pos + 1}),
            width="stretch",
            key="gd-next",
        )


# --------------------------------------------------------------------------- Judge labels
def page_judges() -> None:
    from evalkit.judges import APPLIES_TO, JUDGES, RUBRICS

    trace_store, label_store = stores()
    st.title("Judge labels")
    st.markdown(
        "<p class='ql-lead'>Some qualities are graded by a second model, the judge. Your labels show whether "
        "the judge can be trusted. Read the rubric, then decide pass or fail.</p>",
        unsafe_allow_html=True,
    )
    names = {
        "j1": "Handles unclear questions",
        "j2": "Answer text matches the data",
        "j3": "Refuses unsafe requests well",
    }
    judge = st.segmented_control("Judge", list(names), format_func=lambda j: names[j], default="j1")
    if not judge:
        return
    runs = [p.name for p in runs_list() if (p / "items.jsonl").exists()]
    if not runs:
        st.info("No runs yet. Run a test with judges turned on.")
        return
    run = st.selectbox("Run", runs, key="jd-run")
    items = [i for i in load_items(RESULTS / run) if i["set"] in APPLIES_TO[judge] and i["repeat"] == 0]
    labelled = {a["trace_id"]: a for a in label_store.annotations(task=f"judge:{judge}")}
    n = len(labelled)
    st.progress(min(1.0, n / sum(JUDGE_SPLITS)), text=f"{n} of {sum(JUDGE_SPLITS)} labels for this judge")
    todo = [i for i in items if i["trace_id"] not in labelled]
    if not todo or n >= sum(JUDGE_SPLITS):
        st.success("Nothing left to label for this judge in this run.")
        return
    split = "train" if n < JUDGE_SPLITS[0] else ("dev" if n < sum(JUDGE_SPLITS[:2]) else "test")
    it = todo[0]
    with st.expander("Rubric", icon=":material/menu_book:"):
        st.markdown((RUBRICS / JUDGES[judge]).read_text(encoding="utf-8"))
    t = trace_store.trace(it["trace_id"]) or {}
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(f"### {t.get('question', '')}")
        render_answer(t)
        with st.expander("Steps"):
            render_steps(trace_store, it["trace_id"])
    with right:
        st.caption(f"This label goes to the {split} set of the judge.")
        note = st.text_area("Why?", key=f"jd-note-{it['trace_id']}", height=110)

        def save(verdict: str) -> None:
            if who():
                label_store.add_annotation(
                    it["trace_id"], who(), f"judge:{judge}", verdict, note, [], split=split
                )

        b1, b2 = st.columns(2)
        b1.button(
            "Pass", on_click=save, args=("PASS",), width="stretch", key="jd-pass", icon=":material/check:"
        )
        b2.button(
            "Fail", on_click=save, args=("FAIL",), width="stretch", key="jd-fail", icon=":material/close:"
        )
        need_initials()
    keyboard()


# --------------------------------------------------------------------------- Mistakes summary
def page_summary() -> None:
    _, label_store = stores()
    st.title("Mistakes summary")
    ann = label_store.annotations(task="open_coding")
    if not ann:
        st.info("No labels yet. Label answers on Review answers first.")
        st.page_link(P["page_review"], label="Go to review answers", icon=":material/rate_review:")
        return
    v = Counter(a["verdict"] for a in ann)
    c1, c2, c3 = st.columns(3)
    c1.metric("Labelled", len(ann))
    c2.metric("Passed", v.get("PASS", 0))
    c3.metric("Failed", v.get("FAIL", 0))
    modes = Counter(m for a in ann if a["verdict"] == "FAIL" for m in a["failure_modes"])
    if modes:
        st.subheader("Mistake types among failures")
        df = pd.DataFrame(
            [{"Mistake": f"{k}: {MODE_NAMES.get(k, k)}", "Count": n} for k, n in modes.most_common()]
        )
        st.bar_chart(df, x="Mistake", y="Count", horizontal=True, color="#0B6E4F")
    st.subheader("All notes")
    df = pd.DataFrame(ann)[["labeller", "verdict", "critique", "failure_modes"]]
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button(
        "Download as CSV", pd.DataFrame(ann).to_csv(index=False), "labels.csv", icon=":material/download:"
    )


# --------------------------------------------------------------------------- Browse data
def page_data() -> None:
    from bazaar_world.mysqldb import db_name, fetchall, role_connect
    from querylens.retriever import load_dictionary

    st.title("Browse data")
    st.markdown(
        "<p class='ql-lead'>A read-only look at the BazaarDW warehouse in MySQL, exactly as each role sees it.</p>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    role = c1.segmented_control("Role", ["analyst", "support", "finance"], default="analyst")
    size = c2.segmented_control("Data size", ["full", "small"], default="full")
    if not role or not size:
        return
    try:
        con = role_connect(size, role)
    except Exception as e:  # noqa: BLE001 - explain connection problems in plain words
        st.error(
            f"Cannot connect to MySQL as the {role} user: {e}. Is MySQL running, and was the warehouse built?"
        )
        return
    st.caption(f"MySQL database {db_name(size, role)}, connected as read-only user ql_{role}.")
    try:
        tables = [r[0] for r in fetchall(con, "SHOW FULL TABLES")]
        d = load_dictionary()["tables"]
        table = st.selectbox(
            "Table", tables, format_func=lambda t: f"{t}: {d.get(t, {}).get('description', '')}"
        )
        n = fetchall(con, f"SELECT COUNT(*) FROM `{table}`")[0][0]
        st.metric("Rows", f"{n:,}")
        for note in d.get(table, {}).get("notes", []):
            st.caption(f"Note: {note}")
        cols = fetchall(
            con,
            "SELECT COLUMN_NAME, COLUMN_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = %s "
            "AND TABLE_NAME = %s ORDER BY ORDINAL_POSITION",
            (db_name(size, role), table),
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Column": c,
                        "Type": t,
                        "Meaning": d.get(table, {}).get("columns", {}).get(c, {}).get("description", ""),
                    }
                    for c, t in cols
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**First 100 rows**")
        with con.cursor() as cur:
            cur.execute(f"SELECT * FROM `{table}` LIMIT 100")
            names = [x[0] for x in cur.description or []]
            st.dataframe(pd.DataFrame(list(cur.fetchall()), columns=names), width="stretch", hide_index=True)
    finally:
        con.close()


# --------------------------------------------------------------------------- Help
def page_help() -> None:
    st.title("Help")
    st.markdown("""
**What is this?** QueryLens is an analyst for a fictional Bangladeshi marketplace. It answers questions by writing
SQL over a synthetic warehouse, and a test bench measures how often it gets the right answer.

**A good first session**
1. *Ask*: try a few questions and open “How the analyst worked this out”.
2. *Test runs*: start the quick check (60 questions) and wait for it to finish.
3. *Review answers*: mark 20 to 30 answers pass or fail with a short note.
4. *Check gold*: verify the dev items.

**Words used here**
- *Gold*: the reference SQL for a test question. A result is right when it matches the gold result.
- *Execution accuracy*: share of questions whose result matches the gold. The target is 70%.
- *Unsafe access attempt*: the analyst tried to reach data its role may not see. The target is 0, even if it was blocked.
- *Judge*: a second model that grades answer quality; your labels check whether it agrees with you.
- *Split*: train and dev are for improving the analyst; test is kept aside for the final measurement.

**Where things are**
- Warehouse: MySQL database `bazaar_full` (all columns) and one database per role, such as `bazaar_full_analyst`. Open them in MySQL Workbench.
- Run results: `results/<run name>/`. Your labels: `datasets/labels/annotations.sqlite`.
- Full guide: `docs/QueryLens_Run_Guide.docx`.

**Something went wrong?**
- *No model available*: add a free key to `.env` and reload.
- *A run stopped early*: a free daily limit was reached. Press Resume on Test runs later.
- *Slow answers*: free tiers are rate-limited; Gemini Flash-Lite is the fastest option.
""")


# --------------------------------------------------------------------------- app
P: dict = {}


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.text_input(
            "Your initials",
            key="labeller",
            placeholder="e.g. SH",
            help="Saved with every label and verification.",
        )
    spec = [
        ("Use", page_home, "Home", ":material/home:"),
        ("Use", page_ask, "Ask", ":material/chat:"),
        ("Use", page_runs, "Test runs", ":material/play_circle:"),
        ("Review", page_review, "Review answers", ":material/rate_review:"),
        ("Review", page_gold, "Check gold", ":material/verified:"),
        ("Review", page_judges, "Judge labels", ":material/gavel:"),
        ("Review", page_summary, "Mistakes summary", ":material/bar_chart:"),
        ("Reference", page_data, "Browse data", ":material/table_view:"),
        ("Reference", page_help, "Help", ":material/help:"),
    ]
    pages: dict[str, list] = {}
    for group, fn, title, icon in spec:
        url = "home" if fn is page_home else title.lower().replace(" ", "-")
        pg = st.Page(
            fn, title=title, icon=icon, default=fn is page_home, url_path=None if fn is page_home else url
        )
        P[fn.__name__] = pg
        pages.setdefault(group, []).append(pg)
    st.navigation(pages).run()


main()
