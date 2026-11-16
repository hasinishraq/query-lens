const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType, ShadingType,
  BorderStyle, LevelFormat, AlignmentType, TableOfContents, Footer, PageNumber, PageBreak,
} = require("docx");

const OUT = process.argv[2];
const FONT = "Calibri";
const MONO = "Consolas";
const W = 9360; // content width (DXA) on US Letter with 1" margins

const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120 }, ...opts, children: runs(text) });
function runs(text) {
  // **bold** and `code` inline markup
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun(text.slice(last, m.index)));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: MONO, size: 20 }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun(text.slice(last)));
  return out;
}
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 }, children: runs(t) });
const step = (t, ref = "steps") => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 }, children: runs(t) });

function code(lines) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: "C8CDD3" };
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: [W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: W, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "F3F5F7", color: "auto" },
      borders: { top: border, bottom: border, left: border, right: border },
      margins: { top: 100, bottom: 100, left: 160, right: 160 },
      children: lines.map((l) => new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: l || " ", font: MONO, size: 18 })] })),
    })] })],
  });
}
const gap = () => new Paragraph({ spacing: { after: 80 }, children: [] });

function table(headers, rows, widths) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: "C8CDD3" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (t, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: head ? { type: ShadingType.CLEAR, fill: "1F3A5F", color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ spacing: { after: 0 }, children: head ? [new TextRun({ text: t, bold: true, color: "FFFFFF", size: 20 })] : runs(t).map((r) => r) })],
  });
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
}
const note = (t) => new Paragraph({
  spacing: { before: 60, after: 160 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: "2E75B6", space: 8 } },
  indent: { left: 200 },
  children: runs(t),
});

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("QueryLens: How to Run the Whole Application")] }),
  p("Text-to-SQL analyst agent and evaluation system on the BazaarDW synthetic warehouse. Free models only, $0 API spend."),
  p("Version: 16 November 2026. Platform: Windows 11 with Git Bash or PowerShell (Linux and macOS work the same way)."),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),

  h1("1. What you will run"),
  p("QueryLens has five parts. You run them in this order:"),
  table(["Part", "What it does", "Main command"], [
    ["BazaarDW generator", "Builds the synthetic DuckDB warehouse (small and full sizes) and role databases", "`python -m bazaar_world.generate`"],
    ["Eval datasets", "930 eval items, gold SQL and gold results", "`evalkit datasets`"],
    ["Agent", "Answers one business question with SQL using free LLMs", "`querylens ask`"],
    ["Evaluation runner", "Runs the agent on a slice of items and scores it", "`evalkit run`"],
    ["Viewer", "Web app to inspect traces, verify gold and label", "`streamlit run app/viewer.py`"],
  ], [2200, 4560, 2600]),
  gap(),
  note("**Tip:** every command below starts with `uv run`, which runs it inside the project's Python environment. On Windows there is no `make`, so this guide always shows the full `uv run` command. If you install GNU make, the short `make` targets in the Makefile do the same thing."),

  h1("2. Prerequisites (one time)"),
  table(["Tool", "Version used", "Check with"], [
    ["Python", "3.11 or newer (3.12 tested)", "`python --version`"],
    ["uv (package manager)", "0.8 or newer", "`uv --version`"],
    ["Git", "any recent", "`git --version`"],
    ["Node.js (only for red team)", "20 or newer", "`node --version`"],
    ["Ollama (optional, local model)", "any", "`ollama --version`"],
  ], [3000, 3200, 3160]),
  gap(),
  p("Install uv if it is missing:"),
  code(["pip install uv"]),
  gap(),

  h1("3. Set up the project"),
  step("Open a terminal in the project folder:"),
  code(["cd \"G:\\Personal Projects\\QueryLens_Project\\QueryLens_Project\""]),
  gap(),
  step("Install all Python dependencies (pinned in uv.lock):"),
  code(["uv sync", "", "# optional: embeddings for the hybrid schema retriever (large download)", "uv sync --extra embeddings"]),
  gap(),
  step("Create the `.env` file with your free API keys. Copy `.env.example` to `.env` and fill in the keys you have. A missing key only disables that provider."),
  code(["MISTRAL_API_KEY=", "GEMINI_API_KEY=<your Google AI Studio key>", "GROQ_API_KEY=<your Groq key>",
        "OPENROUTER_API_KEY=<your OpenRouter key>", "OLLAMA_BASE_URL=http://localhost:11434/v1"]),
  note("**Security:** `.env` is git-ignored and must never be committed. Only use free-tier keys; never enable billing (project rule 9). If a key was ever shared in a chat or email, rotate it in the provider's console."),
  step("Check which free models each provider serves today:"),
  code(["uv run querylens models-check", "uv run querylens models-check --verbose   # also lists every served model"]),
  p("A line marked `GONE` means a configured model was removed; update `configs/providers.yaml` with a current free model id."),

  h1("4. Build the warehouse (BazaarDW)"),
  p("Build both sizes. Each build writes the base DuckDB file, three role databases (analyst, support, finance), Parquet exports, table hashes and the planted-fact values."),
  code(["uv run python -m bazaar_world.generate --size small --seed 42 --out data/", "uv run python -m bazaar_world.generate --size full  --seed 42 --out data/"]),
  gap(),
  table(["Size", "Rows", "Build time (measured)", "Use"], [
    ["small", "102,266", "about 4 s", "unit tests, quick experiments"],
    ["full", "995,786", "about 16 s", "reported numbers, evals"],
  ], [1600, 2000, 2600, 3160]),
  gap(),
  p("Check the build (row counts, hashes, the 8 planted facts) and the data dictionary:"),
  code(["uv run python -m bazaar_world.check --size small", "uv run python -m bazaar_world.check --size full",
        "uv run python -m bazaar_world.dictionary_check --size full"]),
  p("Each check prints `OK` or lists the errors."),

  h1("5. Eval datasets and gold results"),
  p("The 930 eval items are already in `datasets/`. Check the composition rules and (re)build the gold results for both snapshots:"),
  code(["uv run evalkit datasets report     # must end with: datasets report: OK",
        "uv run evalkit datasets gold       # runs every gold, alt and trap SQL on small and full"]),
  p("Only rebuild the items themselves if the templates change (this keeps human verifications whose SQL is unchanged):"),
  code(["uv run evalkit datasets build"]),
  gap(),

  h1("6. Ask the agent one question"),
  p("Pick a config from `configs/`. With the keys you have now, use `e_gemini_lite`:"),
  code(["uv run querylens ask \"net revenue by division in 2026 Q2\" --size full --config e_gemini_lite"]),
  p("The output shows the trace id, status, models used, number of calls, tokens, latency, the answer and the SQL."),
  table(["Config", "Agent model", "Judge model", "Needs"], [
    ["e_gemini_lite", "Gemini 3.5 Flash-Lite", "Groq gpt-oss-120b", "Gemini + Groq keys"],
    ["f_groq", "Groq gpt-oss-120b", "Gemini 3.5 Flash-Lite", "Groq + Gemini keys (slow: 8K tokens/min)"],
    ["g_openrouter", "Nemotron 3 Ultra (:free)", "Gemini 3.5 Flash-Lite", "OpenRouter key (50 requests/day)"],
    ["d_gemini", "Gemini Flash", "Groq gpt-oss-120b", "Gemini + Groq keys"],
    ["c_cascade", "Flash-Lite, escalates to Flash", "Groq gpt-oss-120b", "Gemini + Groq keys"],
    ["b_mistral", "Mistral small", "Gemini Flash", "Mistral key (not set yet)"],
    ["a_local", "Ollama qwen2.5:7b", "Gemini Flash", "Ollama installed + model pulled"],
  ], [1800, 2700, 2300, 2560]),
  gap(),
  p("Other agent options: `--role analyst|support|finance` and `--size small|full`. See today's quota use with:"),
  code(["uv run querylens quota"]),
  gap(),

  h1("7. Run an evaluation"),
  p("The runner asks the agent every item in a slice, scores it with the code checks, and writes `results/<run_id>/` (run.json, items.jsonl, summary.json, report.md)."),
  code(["# 60 fixed dev items (fast first check)",
        "uv run evalkit run --slice ci  --size full --config e_gemini_lite --run-id my-first-run", "",
        "# the whole dev split (342 items), with LLM judges J1-J3",
        "uv run evalkit run --slice dev --size full --config e_gemini_lite --judges", "",
        "# 3 repeats for pass^3",
        "uv run evalkit run --slice dev --size full --config e_gemini_lite --repeats 3"]),
  gap(),
  table(["Slice", "Items"], [
    ["ci", "60 fixed dev golden items"], ["train / dev", "162 / 342 items"], ["adv", "150 adversarial items"],
    ["golden:dev, ambiguous:train, ...", "one set and split"], ["ids:gld-0001,gld-0002", "specific items"],
    ["test", "426 items. Refused unless you add `--frozen`; log the run in `docs/test_runs.md`"],
  ], [3600, 5760]),
  gap(),
  h2("7.1 If a free quota runs out"),
  p("Each finished item is saved immediately. When a daily limit is hit the run stops cleanly. Continue later (even the next day) with:"),
  code(["uv run evalkit run --resume my-first-run"]),
  h2("7.2 Report and CI gate"),
  code(["uv run evalkit report --run my-first-run          # rewrites report.md and report.html",
        "uv run evalkit gate   --run my-first-run          # PASS/FAIL against configs/ci.yaml",
        "uv run evalkit gate   --run new-run --baseline my-first-run --markdown gate.md"]),
  h2("7.3 Compare configs (frontier plot)"),
  code(["uv run evalkit frontier --slice ci --size full", "# writes results/published/frontier_ci_full.png and .md"]),

  h1("8. Open the viewer (web app)"),
  code(["uv run streamlit run app/viewer.py --server.port 8501"]),
  p("Then open http://localhost:8501 in a browser. Type your initials in the sidebar first; nothing is saved without them."),
  table(["Tab", "What you do there"], [
    ["Traces", "Pick a run, step through each trace (question, tool steps, SQL, rows, answer). Mark PASS/FAIL, write a critique, tag failure modes. Keys: P pass, F fail, N next, B back."],
    ["Gold review", "Check each item's gold SQL and result. Click Verify to set verified_by, or Flag problem with a note. Only verified items count in reported metrics."],
    ["Judge labels", "Label up to 200 items per judge (60 train, 60 dev, 80 test) for judge validation."],
    ["Error analysis", "See counts of your labels and failure modes; export CSV."],
  ], [1800, 7560]),
  note("**Privacy:** Streamlit listens on all network interfaces, so other devices on your network can reach the viewer. Stop it with Ctrl+C when you are done."),
  p("After labelling judge items, compute judge agreement (TPR, TNR, kappa) for a run that used `--judges`:"),
  code(["uv run evalkit align --run <run_id> --judge j2 --split dev"]),

  h1("9. Error analysis on the scenario pool"),
  p("Run the agent on 300 of the 1,200 scenario questions (no test-split questions) and label the traces in the viewer:"),
  code(["uv run evalkit scenarios --config e_gemini_lite --limit 300 --run-id scenarios-1"]),
  p("Write your taxonomy and counts in `docs/error_analysis.md`."),

  h1("10. Red team (optional, needs Node.js)"),
  code(["cd redteam", "# Command Prompt:", "set PROMPTFOO_PYTHON=..\\.venv\\Scripts\\python.exe",
        "# PowerShell:", "$env:PROMPTFOO_PYTHON = \"..\\.venv\\Scripts\\python.exe\"",
        "npx --yes promptfoo@0.125.1 redteam run -c promptfooconfig.yaml", "cd ..",
        "uv run evalkit run --slice adv --size full --config e_gemini_lite"]),
  gap(),

  h1("11. Tests and code quality"),
  code(["uv run pytest -q                 # 321 tests (full-size tests run if the full DB exists)",
        "uv run ruff check . && uv run ruff format --check .",
        "uv run pyright"]),
  gap(),

  h1("12. Quick start (everything in order)"),
  code(["uv sync",
        "copy .env.example .env      # then add your keys",
        "uv run querylens models-check",
        "uv run python -m bazaar_world.generate --size small --out data/",
        "uv run python -m bazaar_world.generate --size full  --out data/",
        "uv run evalkit datasets gold",
        "uv run querylens ask \"net revenue by division in 2026 Q2\" --size full --config e_gemini_lite",
        "uv run evalkit run --slice ci --size full --config e_gemini_lite --run-id first",
        "uv run streamlit run app/viewer.py"]),
  gap(),

  h1("13. Troubleshooting"),
  table(["Symptom", "Cause and fix"], [
    ["status: error (no API key)", "The config's provider has no key in `.env`. Add the key or pick another config."],
    ["HTTP 400 from Gemini", "Gemini rejects some OpenAI parameters. The client already drops `seed` and keeps thought signatures; clear `.cache/llm` if an old cached reply is involved."],
    ["HTTP 503 / 429", "Provider overloaded or free limit reached. The client retries, then falls back to the next model in the config. Wait or switch config."],
    ["Run stopped: quota", "Daily free limit used up. Resume later with `evalkit run --resume <run_id>`."],
    ["'rate limit: waiting Ns' in the log", "Normal: the limiter waits instead of failing. Real limits are in `configs/providers.yaml`; update them from each console."],
    ["Viewer shows no traces", "Run `querylens ask` or `evalkit run` first, then refresh the page."],
    ["make: command not found", "Use the `uv run ...` commands in this guide, or install GNU make (`choco install make`)."],
    ["Full-size tests skipped", "Build the full warehouse first (section 4)."],
  ], [3000, 6360]),
  gap(),

  h1("14. Project rules to remember"),
  bullet("The agent database is read-only; permissions are enforced by role databases and the SQL guard."),
  bullet("Never tune on the test split; frozen test runs are requested by the human and logged in `docs/test_runs.md`."),
  bullet("Gold SQL and labels are approved only by the human, in the viewer."),
  bullet("Report numbers only from files in `results/`."),
  bullet("Free models only: $0 API spend."),
];

const doc = new Document({
  creator: "QueryLens",
  title: "QueryLens: How to Run the Whole Application",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 44, bold: true, color: "1F3A5F" }, paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 30, bold: true, color: "1F3A5F" }, paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 25, bold: true, color: "2E75B6" }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
  ] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "QueryLens run guide  |  page ", size: 18, color: "666666" }), new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "666666" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("written", OUT); });
