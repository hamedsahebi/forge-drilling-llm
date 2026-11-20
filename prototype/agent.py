"""Tool-using LLM agent that answers engineering questions over the extracted report database.

    python agent.py "Which BHA components were used in well 56-32 and at what depths?"
    python agent.py --demo        # runs the prototype question set, saves outputs/agent_demo.json
    python agent.py --checks      # one question per prompt rule, saves outputs/agent_checks.json
    python agent.py --questions [--verify] [--think] [Q01 ...]   # question set -> outputs/agent_questions/<setting>.json
--verify adds a checker call that reviews each answer against the tool results and sends problems back once.
Without --verify and --think the agent is the same as git tag agent-v1-before-verify.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time

from build_db import DB, OUTD, TABLES
from llm import TEXT_MODEL, chat

def _grounding() -> str:
    """Schema + real category values + one sample row per table (text-to-SQL grounding)."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out = []
    for t, cols in TABLES.items():
        cur = con.execute(f"SELECT * FROM {t} LIMIT 1")
        sample = dict(zip([c[0] for c in cur.description], cur.fetchone() or []))
        sample = {k: (v[:60] if isinstance(v, str) else v) for k, v in sample.items() if k not in ("image_path",)}
        out.append(f"{t}({cols})\n  sample row: {json.dumps(sample, default=str)}")
    for t, c in [("reports", "well"), ("operations", "activity"), ("events", "event_type"), ("figures", "kind"), ("figures", "value_source"),
                 ("chart_readings", "variable"), ("chart_readings", "kind"), ("trust", "tbl"), ("trust", "verdict"), ("trust", "source")]:
        vals = [r[0] for r in con.execute(f"SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL")]
        out.append(f"{t}.{c} values: {vals}")
    return "\n".join(out)


SYSTEM = f"""You are a drilling-engineering assistant. You answer questions using ONLY data extracted
from public Utah FORGE drilling reports, stored in SQLite:
{_grounding()}

Rules:
- Always call a tool before answering. The sample rows above only show the format; they are never an answer.
- Select the columns that say what a value is (variable, kind, field, report_id, page) together with the value,
  and give each number with its own label. Never fill a missing value from another row: say it is not stated.
- Use only the category values listed above in WHERE clauses.
- A hole section (e.g. 8-3/4 in. = 8.75) is identified by reports.hole_size_in; join on report_id. The report's
  hole size is reports.hole_size_in (trust tbl 'depth', field 'hole_size_in'), not a BHA or bit size.
- Before giving a value from a daily report, look up its trust verdict and source and give them with the value, e.g.
  SELECT r.hole_size_in, t.verdict, t.source, t.failed_checks FROM reports r JOIN trust t ON t.report_id = r.report_id
  AND t.tbl = 'depth' AND t.field = 'hole_size_in' WHERE r.report_id = '...'  (table rows also match t.row).
- If a query returns no rows or fewer than expected, inspect the data (SELECT DISTINCT, LIKE '%...%',
  or drop filters) and try again before concluding that nothing exists.
- Tables named eowr_* hold what the End-of-Well Report says about the whole well; pages is the source page range.
  casing, bits and events hold what daily reports say. Both can describe the same item, so never add their counts.
- Use run_sql for lessons, events, casing and bits. For bits, query both bits and eowr_bits.
  Give every event_type a query returns, exactly as returned; for "problems" do not filter types out.
- search_figures takes one or two words (e.g. "MSE", "bit #6"), never a sentence, and returns at most 10 figures;
  to list or count figures use run_sql on figures with LIKE. get_figure(figure_id) returns one figure in full,
  with its whole transcription. Only say a figure was used if a tool call returned it.
- Bit serial numbers are not in bits or eowr_bits. They are printed on the BHA figures as "S/No": search_figures
  for the bit, then get_figure and read the S/No of the bit row. Search figures also for other values a figure could
  print, such as a BHA run date.
- For drilling parameters or depth at a time on well 78B-32, use chart_readings, never figures.key_values. It is read
  from the PDF drawing of the Pason chart without a model, every 5 min; time is 'YYYY-MM-DDTHH:MM:SS'. Cite page and time.
  kind 'reading' is a value on the curve. 'at_max' means the true value is above scale_max and unknown (value is NULL):
  say "above scale_max". 'at_min' means at or near zero. 'text' is a printed label (bit_depth_ft, tvd_ft).
  MAX and MIN are of 5-min samples, not true extremes; if the period has at_max rows, the true maximum is above scale_max.
- figures.value_source says how far a figure's numbers can be trusted. All were read by a vision model and not checked.
  transcription is the text the model copied from the figure; its numbers have the same trust as the figure's value_source.
  'label': printed labels on a diagram or table, in key_values. 'chart': read off a curve or log, in unverified_values;
  these are often wrong, so never give one as a measurement: say it is an unverified vision reading. 'replaced': use chart_readings.
- Table trust has one verdict per daily-report value (report_id, tbl, field). Never give a value with verdict 'fail'
  as fact; say the check failed. 'pdf_read' = copied from the PDF by code (source table, code or column), not by the
  model; 'pass' = confirmed by a check; 'weak' or 'unchecked' with source 'llm' = a model value nobody checked: say so.
- For NPT use operations.is_npt. npt_source tells where it came from: "column" = the operator's Non-Prod mark,
  "llm" = a model estimate, which you must say. Do not use llm_is_npt. Group NPT answers by npt_source.
- Glossary: MSE = mechanical specific energy; NPT = non-productive time; BHA = bottom-hole assembly;
  ROP = rate of penetration; WOB = weight on bit; DDR = daily drilling report; HWDP = heavy-weight drill pipe.
Answer concisely with numbers, and cite report_id or figure_id."""

TOOLS = [
    {"type": "function", "function": {
        "name": "run_sql", "description": "Run one read-only SQLite SELECT query. Returns at most 50 rows.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_figures", "description": "Find figures whose caption, title, description, values or transcribed text contain a keyword.",
        "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}}},
    {"type": "function", "function": {
        "name": "get_figure", "description": "Return one figure in full (values, value_source, whole transcription) by figure_id.",
        "parameters": {"type": "object", "properties": {"figure_id": {"type": "string"}}, "required": ["figure_id"]}}},
]


def run_sql(query: str) -> str:
    if not query.lstrip().lower().startswith(("select", "with")):
        return "error: only SELECT queries are allowed"
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        cur = con.execute(query)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchmany(50)
        out = json.dumps({"columns": cols, "rows": rows}, default=str)
        total = len(rows)
        while len(out) > 6000:  # keep whole rows, and say so, so the model does not take a part for the whole
            rows.pop()
            out = json.dumps({"columns": cols, "rows": rows, "note": f"only {len(rows)} of {total} rows fit; narrow the query"}, default=str)
        return out
    except sqlite3.Error as e:
        return f"sql error: {e}"


TRUST_NOTE = {  # repeated in the tool result: the rule in SYSTEM alone was not followed
    "chart": "numbers read off a curve by a vision model, not checked: give them only as unverified vision readings",
    "label": "printed labels copied by a vision model, not checked: say the value is a vision reading",
    "replaced": "use chart_readings for these numbers",
}


def get_figure(figure_id: str) -> str:
    fid = figure_id.replace(chr(39), "")
    out = run_sql("SELECT figure_id, well, page, kind, caption, value_source, key_values, unverified_values, description, "
                  f"transcription FROM figures WHERE figure_id = '{fid}'")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    src = con.execute("SELECT value_source FROM figures WHERE figure_id = ?", (fid,)).fetchone()
    return out + (f"\nTRUST: {TRUST_NOTE[src[0]]}." if src and src[0] in TRUST_NOTE else "")


STOPWORDS = {"the", "and", "for", "with", "from", "used", "show", "shows", "figure", "figures", "well", "section"}
FIGURE_ID = re.compile(r"\bD\d\d_p\d+_\d+\b")


def search_figures(keyword: str) -> str:
    if m := FIGURE_ID.search(keyword):  # the model passes a sentence that names the figure
        return get_figure(m[0])
    k = keyword.replace(chr(39), "")
    named = " OR ".join(f"{c} LIKE '%{k}%'" for c in ("caption", "title", "description"))
    # Only the text around the match: ten full transcriptions could use up the tool's 6,000 characters
    out = run_sql(f"SELECT figure_id, well, page, kind, caption, value_source, key_values, unverified_values, description, "
                  f"CASE WHEN transcription LIKE '%{k}%' THEN substr(transcription, "
                  f"max(instr(lower(transcription), lower('{k}')) - 150, 1), 300) END AS transcription_match FROM figures "
                  f"WHERE {named} OR key_values LIKE '%{k}%' OR unverified_values LIKE '%{k}%' OR transcription LIKE '%{k}%' "
                  f"ORDER BY ({named}) DESC LIMIT 10")
    words = [w for w in re.findall(r"[\w/-]{3,}", k) if w.lower() not in STOPWORDS]
    if '"rows": []' in out and len(words) > 1:  # a sentence matches nothing: require every word in caption, well or kind
        text = "(coalesce(caption, '') || ' ' || coalesce(title, '') || ' ' || well || ' ' || kind)"
        cond = " AND ".join(f"{text} LIKE '%{w}%'" for w in words)
        out = run_sql(f"SELECT figure_id, well, page, kind, caption, value_source FROM figures WHERE {cond} LIMIT 10")
    return out + "\nUse get_figure(figure_id) for a figure's full transcription."


FUNCS = {"run_sql": run_sql, "search_figures": search_figures, "get_figure": get_figure}


VERIFY_SYSTEM = """You check an answer written by a drilling-data agent. You get the question, the tool results the agent saw
and its answer. Report a problem only when one of these is true:
1. A number in the answer is not in the tool results and cannot be computed from them.
2. The answer states a value as fact although the tool results mark it with verdict fail or conflict, or as an unverified
   vision reading, or as at_max / at_min (an axis limit, not a reading).
3. The answer misses part of the question, or a comparison or conclusion does not follow from the numbers.
4. The tool results are empty or do not cover the question, yet the answer gives a value instead of saying it is not available.
5. The question assumes something the tool results contradict, and the answer goes along with it.
Do not judge style. If there is no problem, return ok true and an empty list."""
VERIFY_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"},
                                                  "problems": {"type": "array", "items": {"type": "string"}}},
                 "required": ["ok", "problems"]}


def verify(question: str, answer: str, trace: list[dict], think: bool = False) -> dict:
    """Second call that checks the answer against the tool results (runbook F2 ablation; off unless --verify)."""
    seen = "\n\n".join(f"{t['tool']}({json.dumps(t['args'])[:300]}):\n{t['result']}" for t in trace)[-12000:]
    user = f"QUESTION:\n{question}\n\nTOOL RESULTS:\n{seen or '(no tool was called)'}\n\nANSWER:\n{answer}"
    r = chat([{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}],
             model=TEXT_MODEL, schema=VERIFY_SCHEMA, num_ctx=16384, think=think)["message"]["content"]
    try:
        out = json.loads(r)
        return {"ok": bool(out.get("ok")), "problems": [str(p) for p in out.get("problems", [])]}
    except json.JSONDecodeError:
        return {"ok": True, "problems": [], "error": r[:300]}  # an unreadable check never blocks the answer


def ask(question: str, max_steps: int = 8, verbose: bool = True, verify_answer: bool = False,
        think: bool = False, max_revisions: int = 1) -> dict:
    """Tool loop; with verify_answer a checker reviews the final answer and its problems go back to the agent
    (at most max_revisions times). Defaults reproduce the agent as tagged agent-v1-before-verify."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    trace, checks, t0 = [], [], time.time()
    first_answer = None
    for _ in range(max_steps):
        msg = chat(messages, model=TEXT_MODEL, tools=TOOLS, num_ctx=16384, think=think)["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = msg.get("content", "").strip()
            if first_answer is None:
                first_answer = answer
            if verify_answer and len(checks) < max_revisions:
                v = verify(question, answer, trace, think=think)
                checks.append(v)
                if verbose:
                    print(f"  check: ok={v['ok']} {v['problems']}")
                if not v["ok"] and v["problems"]:
                    messages.append({"role": "user", "content": "A checker found these problems in your answer:\n- "
                                     + "\n- ".join(v["problems"])
                                     + "\nUse the tools if needed, then give a corrected answer. If a problem is wrong, "
                                       "keep your answer and say why."})
                    continue
            return {"question": question, "answer": answer, "first_answer": first_answer, "checks": checks,
                    "trace": trace, "seconds": round(time.time() - t0, 1)}
        for c in calls:
            fn, args = c["function"]["name"], c["function"]["arguments"]
            result = FUNCS[fn](**args) if fn in FUNCS else f"unknown tool {fn}"
            trace.append({"tool": fn, "args": args, "result": result[:1500]})
            if verbose:
                print(f"  -> {fn}({json.dumps(args)[:200]})")
            messages.append({"role": "tool", "content": result, "tool_name": fn})
    return {"question": question, "answer": "(step limit reached)", "first_answer": first_answer, "checks": checks,
            "trace": trace, "seconds": round(time.time() - t0, 1)}


DEMO = [
    "How many non-productive-time hours were logged per well, and what activities caused them?",
    "List all drilling problems or events (stuck pipe, losses, failures) with well, date and depth.",
    "What bits were run in well 56-32 (size, type, depth in/out) and which had the best ROP?",
    "Summarise the BHA components used in the 8-3/4 in. section of 56-32 using reports and BHA figures.",
    "Which figures show MSE or drilling parameters, and what do they indicate?",
    "What mud density and viscosity ranges were used in each well?",
]

# Rule checks (runbook D2): each question tests one prompt rule; `expect` is what a right answer contains,
# checked by SQL against the database on 2026-11-09.
CHECKS = [
    {"rule": "chart_readings for a parameter at a time",
     "q": "What were the rotary RPM and the hook load on well 78B-32 at 08:00 on 17 July 2021?",
     "expect": "chart_readings, page 127: hook load 39.9 klbs (reading); RPM at_min, i.e. about zero"},
    {"rule": "chart value from a figure is unverified",
     "q": "What on-bottom ROP does figure D08_p68_0 (MSE parameters, 8-3/4 in. section, bit #6) show?",
     "expect": "says it is an unverified vision reading (value_source chart); 1000 ft/hr is listed, likely the axis maximum"},
    {"rule": "serial number from a figure transcription",
     "q": "What is the serial number of bit #6 in well 56-32?",
     "expect": "search_figures -> D08_p47_0 (BHA #6 with bit #6) transcription, S/No A275660; vision reading, not checked"},
    {"rule": "value with a fail verdict",
     "q": "What was the hole size in daily report D07_r005?",
     "expect": "16 in. but the trust check failed (verdict fail, hole_size_sequence), so not given as fact"},
    {"rule": "trust.source says how a value was read",
     "q": "Was the 7 in. casing OD in report D07_r030 read from the PDF by code, or is it a model value?",
     "expect": "trust: verdict pdf_read, source table -> copied from the Casing table by code"},
]

QUESTIONS = OUTD.parent / "data" / "questions" / "questions.json"


def run_questions(verify_answer: bool, think: bool, only: list[str]) -> None:
    """Question set (make_questions.py); one output file per setting, resumable, so the four runs can be compared."""
    tag = "base" + ("_verify" if verify_answer else "") + ("_think" if think else "")
    out = OUTD / "agent_questions" / f"{tag}.json"
    out.parent.mkdir(exist_ok=True)
    done = {r["id"]: r for r in json.loads(out.read_text(encoding="utf-8"))} if out.exists() else {}
    for q in json.loads(QUESTIONS.read_text(encoding="utf-8")):
        if q["id"] in done or (only and q["id"] not in only):
            continue
        print(f"{q['id']} [{q['category']}] {q['question']}")
        r = ask(q["question"], verify_answer=verify_answer, think=think)
        print("A:", r["answer"][:600], f"({r['seconds']}s)\n   gold: {q['answer']}\n")
        done[q["id"]] = {**q, **r, "setting": tag}
        out.write_text(json.dumps(list(done.values()), indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if "--questions" in flags:  # python agent.py --questions [--verify] [--think] [Q01 Q02 ...]
        run_questions("--verify" in flags, "--think" in flags, [a for a in sys.argv[1:] if not a.startswith("--")])
    elif sys.argv[1:] == ["--checks"]:
        results = []
        for c in CHECKS:
            print("Q:", c["q"])
            r = ask(c["q"])
            print("A:", r["answer"], f"({r['seconds']}s)\n   expect: {c['expect']}\n")
            results.append({**c, **r})
        (OUTD / "agent_checks.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    elif sys.argv[1:] == ["--demo"]:
        results = []
        for q in DEMO:
            print("Q:", q)
            r = ask(q)
            print("A:", r["answer"], f"({r['seconds']}s)\n")
            results.append(r)
        (OUTD / "agent_demo.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    else:
        r = ask(" ".join(sys.argv[1:]))
        print(r["answer"])
