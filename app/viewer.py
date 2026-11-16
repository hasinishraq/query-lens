"""QueryLens trace viewer and annotation tool (``make viewer``).

Tabs:
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
        b1.button("PASS (p)", on_click=save, args=("PASS",), width="stretch")
        b2.button("FAIL (f)", on_click=save, args=("FAIL",), width="stretch")
        n1, n2 = st.columns(2)
        n1.button("Back (b)", on_click=lambda: st.session_state.update(pos=pos - 1), width="stretch")
        n2.button("Next (n)", on_click=lambda: st.session_state.update(pos=pos + 1), width="stretch")
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
    if b1.button("Verify", width="stretch"):
        if not who:
            st.error("Enter your initials in the sidebar first.")
        else:
            _save_verification(it.id, who)
            st.session_state["gpos"] = pos + 1
            st.rerun()
    if b2.button("Flag problem", width="stretch"):
        with open(FLAGS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": it.id, "by": who, "note": note}) + "\n")
        st.success("flagged; Claude will not change gold without your approval")
    b3.button("Back", on_click=lambda: st.session_state.update(gpos=pos - 1), width="stretch")
    b4.button("Next", on_click=lambda: st.session_state.update(gpos=pos + 1), width="stretch")


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
    b1.button("PASS (p)", on_click=save, args=("PASS",), width="stretch")
    b2.button("FAIL (f)", on_click=save, args=("FAIL",), width="stretch")
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


def main() -> None:
    trace_store, label_store = stores()
    st.sidebar.title("QueryLens")
    who = labeller()
    t1, t2, t3, t4 = st.tabs(["Traces", "Gold review", "Judge labels", "Error analysis"])
    with t1:
        tab_traces(trace_store, label_store, who)
    with t2:
        tab_gold(who)
    with t3:
        tab_judges(trace_store, label_store, who)
    with t4:
        tab_analysis(label_store)


main()
