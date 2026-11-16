"""QueryLens trace viewer and annotation tool (``make viewer``).

Tabs:
0. **Ask**: ask the agent one question and see its answer, SQL and trace.
0. **Runs**: start, follow and resume evaluation runs; read their reports.
1. **Traces**: one trace per screen (question, steps, SQL, rows, answer); PASS/FAIL, critique, failure-mode tags.
2. **Gold review**: the human verifies gold SQL per item (sets ``verified_by``) or flags a problem.
3. **Judge labels**: human PASS/FAIL labels for judge validation (60 train / 60 dev / 80 test per judge).
4. **Error analysis**: counts of human labels and failure modes; CSV export.

Every label written here comes from the person using the viewer (rule 5). Keyboard: P = pass, F = fail,
N = next, B = back (click once inside the page first).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evalkit.datasets import DATASETS, SET_FILES, gold_sql, load_all, load_gold_hashes, variant_sql  # noqa: E402
from evalkit.report import MODE_NAMES, load_items  # noqa: E402
from evalkit.trace import TraceStore  # noqa: E402

TRACES_DB = ROOT / "data" / "traces.sqlite"
LABELS_DB = DATASETS / "labels" / "annotations.sqlite"
FLAGS = DATASETS / "labels" / "gold_flags.jsonl"
RESULTS = ROOT / "results"
JUDGE_SPLITS = (60, 60, 80)

st.set_page_config(page_title="QueryLens viewer", layout="wide")


@st.cache_resource
def stores() -> tuple[TraceStore, TraceStore]:
    return TraceStore(TRACES_DB), TraceStore(LABELS_DB)


def keyboard() -> None:
    components.html(
        """
    <script>
    const doc = window.parent.document;
    if (!doc.__ql_keys) {
      doc.__ql_keys = true;
      doc.addEventListener('keydown', (e) => {
        if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
        const map = {p: 'PASS', f: 'FAIL', n: 'Next', b: 'Back'};
        const label = map[e.key.toLowerCase()];
        if (!label) return;
        const btn = Array.from(doc.querySelectorAll('button')).find(x => x.innerText.trim().startsWith(label));
        if (btn) btn.click();
      });
    }
    </script>""",
        height=0,
    )


def labeller() -> str:
    return st.sidebar.text_input("Your initials (required to label)", key="labeller").strip()


# --------------------------------------------------------------------------- traces tab
def trace_panel(trace_store: TraceStore, trace_id: str) -> None:
    t = trace_store.trace(trace_id)
    if not t:
        st.warning("trace not found")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("status", t["status"])
    c2.metric("latency (s)", t["latency_s"])
    c3.metric("tool calls", t["n_tool_calls"])
    c4.metric("tokens in/out", f"{t['tokens_in']}/{t['tokens_out']}")
    st.markdown(f"**Question** ({t['role']}): {t['question']}")
    st.caption(
        f"model {t['model']} | models used {t['models_used']} | prompt {t['prompt_hash']} | git {t['git_sha']}"
    )
    for s in trace_store.spans(trace_id):
        a = s["attrs"]
        if s["kind"] == "llm":
            calls = ", ".join(tc["name"] for tc in a.get("tool_calls", [])) or "(text)"
            with st.expander(f"LLM -> {calls}  [{a.get('returned_model')}, {a.get('tokens_in')} in]"):
                if a.get("content"):
                    st.write(a["content"])
                st.json(a.get("tool_calls", []))
        elif s["kind"] == "tool":
            label = f"tool {s['name']}" + (f"  ERROR: {a.get('error')}" if a.get("error") else "")
            with st.expander(label):
                st.json(a.get("args", {}))
                st.code(str(a.get("output_preview", ""))[:3000])
        elif s["kind"] == "guard" and a.get("decision") == "reject":
            st.error(f"guard rejected ({a.get('rule')}): {a.get('message')}")
        elif s["kind"] == "sql":
            st.code(a.get("sql", ""), language="sql")
    if (t.get("attrs") or {}).get("clarifying_question"):
        st.info(f"Agent asked: {t['attrs']['clarifying_question']}")
    st.markdown("**Final answer**")
    st.write(t["final_text"] or "_(none)_")
    if t["final_sql"]:
        st.code(t["final_sql"], language="sql")
    if t["final_rows"]:
        st.dataframe(pd.DataFrame(t["final_rows"]), width="stretch")


def tab_traces(trace_store: TraceStore, label_store: TraceStore, who: str) -> None:
    runs = sorted([p.name for p in RESULTS.glob("*") if (p / "items.jsonl").exists()], reverse=True)
    run = st.selectbox("Run", ["(all traces)"] + runs)
    items_by_trace: dict[str, dict] = {}
    if run != "(all traces)":
        for it in load_items(RESULTS / run):
            items_by_trace[it["trace_id"]] = it
        df = pd.DataFrame(
            [
                {
                    "trace_id": k,
                    "item": v["item_id"],
                    "set": v["set"],
                    "split": v["split"],
                    "difficulty": v["dims"]["difficulty"],
                    "trap": v["dims"]["trap"],
                    "behaviour": v["behaviour"],
                    "exec_match": v.get("exec_match"),
                }
                for k, v in items_by_trace.items()
            ]
        )
    else:
        df = pd.DataFrame(trace_store.list_traces(limit=5000)).rename(columns={"id": "trace_id"})
    if df.empty:
        st.info("No traces yet. Run `make agent` or `make eval`.")
        return
    with st.sidebar:
        st.subheader("Filters")
        for col in ("set", "split", "difficulty", "trap", "behaviour", "exec_match"):
            if col in df.columns:
                vals = sorted(df[col].dropna().astype(str).unique())
                pick = st.multiselect(col, vals)
                if pick:
                    df = df[df[col].astype(str).isin(pick)]
        only_unlabelled = st.checkbox("only unlabelled", value=True)
    done = {a["trace_id"] for a in label_store.annotations(task="open_coding")}
    if only_unlabelled:
        df = df[~df["trace_id"].isin(done)]
    ids = df["trace_id"].tolist()
    st.caption(f"{len(ids)} traces in view; {len(done)} labelled so far")
    if not ids:
        return
    pos = st.session_state.setdefault("pos", 0) % len(ids)
    tid = ids[pos]
    left, right = st.columns([3, 1])
    with left:
        it = items_by_trace.get(tid)
        if it:
            st.caption(
                f"item {it['item_id']} ({it['set']}/{it['split']}) expected {it['expected_type']}, "
                f"behaviour {it['behaviour']}, exec_match {it.get('exec_match')}"
            )
        trace_panel(trace_store, tid)
    with right:
        st.subheader("Label")
        critique = st.text_area("Critique (what went wrong / right)", key=f"crit-{tid}")
        modes = st.multiselect(
            "Failure modes", [f"{k}: {v}" for k, v in MODE_NAMES.items()] + ["other"], key=f"modes-{tid}"
        )
        custom = st.text_input("New open code (optional)", key=f"code-{tid}")
        tags = [m.split(":")[0] for m in modes] + ([custom] if custom else [])

        def save(verdict: str) -> None:
            if not who:
                st.error("Enter your initials in the sidebar first.")
                return
            label_store.add_annotation(tid, who, "open_coding", verdict, critique, tags)
            st.session_state["pos"] = pos + 1

        b1, b2 = st.columns(2)
        b1.button("PASS (p)", on_click=save, args=("PASS",), width="stretch", key="tr-pass")
        b2.button("FAIL (f)", on_click=save, args=("FAIL",), width="stretch", key="tr-fail")
        n1, n2 = st.columns(2)
        n1.button(
            "Back (b)", on_click=lambda: st.session_state.update(pos=pos - 1), width="stretch", key="tr-back"
        )
        n2.button(
            "Next (n)", on_click=lambda: st.session_state.update(pos=pos + 1), width="stretch", key="tr-next"
        )
    keyboard()


# --------------------------------------------------------------------------- gold review tab
def _save_verification(item_id: str, who: str) -> None:
    for name in SET_FILES:
        path = DATASETS / f"{name}.jsonl"
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        changed = False
        for k, line in enumerate(lines):
            if not line.strip():
                continue
            d = json.loads(line)
            if d["id"] == item_id:
                d["verified_by"] = who
                lines[k] = json.dumps(d, ensure_ascii=False)
                changed = True
        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tab_gold(who: str) -> None:
    items = [i for i in load_all() if i.gold_sql_file and i.set in ("golden", "adversarial")]
    c1, c2, c3 = st.columns(3)
    split = c1.selectbox("Split", ["dev", "test", "train", "(all)"])
    status = c2.selectbox("Status", ["unverified", "verified", "(all)"])
    snap = c3.selectbox("Snapshot", ["full", "small"])
    if split != "(all)":
        items = [i for i in items if i.split == split]
    if status != "(all)":
        items = [i for i in items if bool(i.verified_by) == (status == "verified")]
    st.caption(f"{len(items)} items")
    if not items:
        return
    pos = st.session_state.setdefault("gpos", 0) % len(items)
    it = items[pos]
    hashes = load_gold_hashes(snap)
    h = hashes.get(Path(it.gold_sql_file or "").stem, {})
    st.markdown(
        f"### {it.id}  ({it.set}/{it.split}, {it.dims.intent}, {it.dims.difficulty}, trap {it.dims.trap})"
    )
    st.markdown(f"**Question:** {it.question}")
    if it.planted_fact:
        st.caption(f"planted fact {it.planted_fact}")
    st.code(gold_sql(it) or "", language="sql")
    st.caption(
        f"double-written check (alt SQL same fingerprint): {h.get('alt_match')} | rows {h.get('rows')} | "
        f"trap variant differs: {h.get('trap_differs')}"
    )
    with st.expander("alt SQL / trap SQL"):
        st.code(variant_sql(it, "alt") or "-", language="sql")
        st.code(variant_sql(it, "trap") or "-", language="sql")
    pq = DATASETS / "gold_results" / snap / f"{Path(it.gold_sql_file or '').stem}.parquet"
    if pq.exists():
        st.dataframe(
            duckdb.sql(f"SELECT * FROM read_parquet('{pq.as_posix()}') LIMIT 50").df(), width="stretch"
        )
    note = st.text_input("Problem note (if the gold is wrong)", key=f"note-{it.id}")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Verify", width="stretch", key="gold-verify"):
        if not who:
            st.error("Enter your initials in the sidebar first.")
        else:
            _save_verification(it.id, who)
            st.session_state["gpos"] = pos + 1
            st.rerun()
    if b2.button("Flag problem", width="stretch", key="gold-flag"):
        if not who:
            st.error("Enter your initials in the sidebar first.")
        elif not note.strip():
            st.error("Write what is wrong with this gold item first.")
        else:
            with open(FLAGS, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": it.id, "by": who, "note": note}) + "\n")
            st.success("flagged; Claude will not change gold without your approval")
    b3.button(
        "Back", on_click=lambda: st.session_state.update(gpos=pos - 1), width="stretch", key="gold-back"
    )
    b4.button(
        "Next", on_click=lambda: st.session_state.update(gpos=pos + 1), width="stretch", key="gold-next"
    )


# --------------------------------------------------------------------------- judge labels tab
def tab_judges(trace_store: TraceStore, label_store: TraceStore, who: str) -> None:
    from evalkit.judges import APPLIES_TO

    judge = st.selectbox("Judge", ["j1", "j2", "j3"])
    runs = sorted([p.name for p in RESULTS.glob("*") if (p / "items.jsonl").exists()], reverse=True)
    if not runs:
        st.info("No runs yet.")
        return
    run = st.selectbox("Run with traces to label", runs, key="jrun")
    items = [i for i in load_items(RESULTS / run) if i["set"] in APPLIES_TO[judge] and i["repeat"] == 0]
    labelled = {a["trace_id"]: a for a in label_store.annotations(task=f"judge:{judge}")}
    n_done = len(labelled)
    st.caption(f"{n_done} / {sum(JUDGE_SPLITS)} labels for {judge} (60 train / 60 dev / 80 test)")
    todo = [i for i in items if i["trace_id"] not in labelled]
    if not todo or n_done >= sum(JUDGE_SPLITS):
        st.success("Done for this run.")
        return
    split = (
        "train"
        if n_done < JUDGE_SPLITS[0]
        else ("dev" if n_done < JUDGE_SPLITS[0] + JUDGE_SPLITS[1] else "test")
    )
    it = todo[0]
    st.markdown(f"Next label goes to the **{split}** split of {judge}. Judge only what the rubric asks.")
    with st.expander("Rubric", expanded=False):
        from evalkit.judges import JUDGES, RUBRICS

        st.markdown((RUBRICS / JUDGES[judge]).read_text(encoding="utf-8"))
    trace_panel(trace_store, it["trace_id"])
    critique = st.text_area("Critique", key=f"jc-{it['trace_id']}")

    def save(verdict: str) -> None:
        if not who:
            st.error("Enter your initials in the sidebar first.")
            return
        label_store.add_annotation(it["trace_id"], who, f"judge:{judge}", verdict, critique, [], split=split)

    b1, b2 = st.columns(2)
    b1.button("PASS (p)", on_click=save, args=("PASS",), width="stretch", key="jd-pass")
    b2.button("FAIL (f)", on_click=save, args=("FAIL",), width="stretch", key="jd-fail")
    keyboard()


# --------------------------------------------------------------------------- error analysis tab
def tab_analysis(label_store: TraceStore) -> None:
    ann = label_store.annotations(task="open_coding")
    st.metric("open-coding labels", len(ann))
    if not ann:
        return
    v = Counter(a["verdict"] for a in ann)
    st.write(dict(v))
    modes = Counter(m for a in ann if a["verdict"] == "FAIL" for m in a["failure_modes"])
    st.bar_chart(pd.Series(modes).sort_values(ascending=False))
    df = pd.DataFrame(ann)
    st.dataframe(df[["trace_id", "labeller", "verdict", "critique", "failure_modes"]], width="stretch")
    st.download_button("Export CSV", df.to_csv(index=False), "open_coding.csv")


CONFIGS = ROOT / "configs"
UI_SLICES = {
    "ci (60 fixed dev questions)": "ci",
    "dev (342 items, all sets)": "dev",
    "train (162 items)": "train",
    "adversarial (150 attacks)": "adv",
    "golden dev only": "golden:dev",
    "ambiguous dev": "ambiguous:dev",
    "unanswerable dev": "unanswerable:dev",
    "Bangla dev": "bilingual_bn:dev",
    "Banglish dev": "bilingual_banglish:dev",
}


def usable_configs() -> list[str]:
    """Agent configs whose agent provider has a key (or needs none)."""
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
    return out


@st.cache_resource
def get_agent(config: str, role: str, size: str):  # noqa: ANN201
    from querylens.agent import Agent
    from querylens.config import load_config

    return Agent(load_config(config), role=role, size=size, store=stores()[0])


def tab_ask(trace_store: TraceStore) -> None:
    st.subheader("Ask the analyst agent")
    cfgs = usable_configs()
    if not cfgs:
        st.error("No config has an API key. Add keys to `.env` and restart the viewer.")
        return
    c1, c2, c3 = st.columns(3)
    config = c1.selectbox(
        "Model config", cfgs, index=cfgs.index("e_gemini_lite") if "e_gemini_lite" in cfgs else 0
    )
    role = c2.selectbox("Role", ["analyst", "support", "finance"])
    size = c3.selectbox("Warehouse size", ["full", "small"])
    examples = [
        "",
        "What was GMV by division in 2026 Q2?",
        "Top 5 sellers by GMV in 2025",
        "Which courier had the highest late delivery rate in 2025?",
        "Show monthly cancellation rate in 2025",
        "List customer phone numbers in Dhaka",
    ]
    ex = st.selectbox("Example question (optional)", examples)
    q = st.text_area("Your question", value=ex, height=80, key="ask-q")
    if st.button("Ask", type="primary", key="ask-go", disabled=not q.strip()):
        with st.spinner("The agent is working (free models can take 10-60 s)..."):
            res = get_agent(config, role, size).run(q.strip(), run_id="ui-ask")
        st.session_state["ask-trace"] = res.trace_id
        st.session_state["ask-res"] = res.model_dump()
    r = st.session_state.get("ask-res")
    if r:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("status", r["status"])
        m2.metric("LLM calls", r["n_llm_calls"])
        m3.metric("tokens in", r["tokens_in"])
        m4.metric("seconds", r["latency_s"])
        if r["clarifying_question"]:
            st.info(f"The agent asks: {r['clarifying_question']}")
        if r["final_text"]:
            st.success(r["final_text"])
        elif r["status"] == "error":
            st.error(r["stop_reason"])
        if r["final_sql"]:
            st.code(r["final_sql"], language="sql")
        if r["final_rows"]:
            st.dataframe(pd.DataFrame(r["final_rows"]), width="stretch")
        with st.expander("Full trace (every step)"):
            trace_panel(trace_store, st.session_state["ask-trace"])


def _run_progress(run_dir: Path) -> tuple[int, int]:
    done = len(load_items(run_dir))
    info = run_dir / "run.json"
    planned = 0
    if info.exists():
        d = json.loads(info.read_text(encoding="utf-8"))
        planned = len(d.get("item_ids", [])) * int(d.get("repeats", 1))
    return done, planned


def _launch(args: list[str], run_id: str) -> None:
    import subprocess

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


def tab_runs() -> None:
    st.subheader("Start an evaluation run")
    cfgs = usable_configs()
    c1, c2, c3, c4 = st.columns(4)
    slice_label = c1.selectbox("Questions", list(UI_SLICES))
    config = c2.selectbox("Model config", cfgs, key="run-cfg")
    limit = c3.number_input("Max items (0 = all)", min_value=0, value=0, step=10)
    repeats = c4.number_input("Repeats", min_value=1, max_value=3, value=1)
    judges = st.checkbox("Also run LLM judges (uses more free quota)")
    st.caption(
        "The test split is not offered here: frozen test runs are requested deliberately and logged "
        "(`uv run evalkit run --slice test --frozen`)."
    )
    default_id = (
        f"ui-{config}-{UI_SLICES[slice_label].replace(':', '_')}-{pd.Timestamp.now().strftime('%m%d-%H%M')}"
    )
    run_id = st.text_input("Run name", value=default_id)
    if st.button("Start run", type="primary", key="run-start"):
        if (RESULTS / run_id / "run.json").exists():
            st.error("A run with that name exists; use Resume below or pick another name.")
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
            _launch(args, run_id)
            st.success(f"Started {run_id}. It runs in the background; this page shows its progress.")
    st.divider()
    st.subheader("Runs")
    runs = sorted(
        [p for p in RESULTS.glob("*") if (p / "run.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        st.info("No runs yet.")
        return
    rows = []
    for p in runs:
        done, planned = _run_progress(p)
        acc = None
        if (p / "summary.json").exists():
            s = json.loads((p / "summary.json").read_text(encoding="utf-8"))
            acc = (s.get("exec_accuracy") or {}).get("rate")
        rows.append(
            {
                "run": p.name,
                "done": done,
                "planned": planned,
                "exec accuracy": None if acc is None else round(acc, 3),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    sel = st.selectbox("Open run", [r["run"] for r in rows])
    run_dir = RESULTS / sel
    done, planned = _run_progress(run_dir)
    st.progress(min(1.0, done / planned) if planned else 0.0, text=f"{done}/{planned} items")
    b1, b2 = st.columns(2)
    if b1.button("Refresh", key="run-refresh"):
        st.rerun()
    if b2.button("Resume / finish this run", key="run-resume", disabled=planned and done >= planned):
        _launch(["--resume", sel], sel)
        st.success("Resumed in the background.")
    if (run_dir / "report.md").exists():
        st.markdown((run_dir / "report.md").read_text(encoding="utf-8"))
    else:
        st.caption("The report appears when the run finishes (or stops on a quota limit).")
    log = run_dir / "ui_run.log"
    if log.exists():
        with st.expander("Run log (last lines)"):
            lines = [
                x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if "httpx" not in x
            ]
            st.code("\n".join(lines[-30:]) or "(empty)")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    trace_store, label_store = stores()
    st.sidebar.title("QueryLens")
    who = labeller()
    t0, tr, t1, t2, t3, t4 = st.tabs(
        ["Ask", "Runs", "Traces", "Gold review", "Judge labels", "Error analysis"]
    )
    with t0:
        tab_ask(trace_store)
    with tr:
        tab_runs()
    with t1:
        tab_traces(trace_store, label_store, who)
    with t2:
        tab_gold(who)
    with t3:
        tab_judges(trace_store, label_store, who)
    with t4:
        tab_analysis(label_store)


main()
