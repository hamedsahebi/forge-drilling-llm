# From PDF drilling reports to a trusted database

Code and data for a stage-by-stage evaluation of **local language models** that turn public drilling PDFs into a queryable database. The documents come from three Utah FORGE geothermal wells.

Each stage is scored against a reference that the pipeline did not produce:

| Stage | Reference |
|---|---|
| Table and time-log fields | Labels by a drilling engineer, including 6 reports labelled blind from empty fields |
| Chart values | The Pason rig-sensor records the chart was drawn from |
| Non-productive-time (NPT) flag | The operator's own Non-Prod marks |
| Agent answers | 52 engineering questions, graded by the engineer |

All models run locally through [Ollama](https://ollama.com) on a laptop with an 8 GB GPU: `qwen3:8b` for text and `qwen2.5vl:7b` for figures. No data leaves the machine.

> The paper describing this work is under review. Please cite it (see `CITATION.cff`) once published.
> The earlier system this work validates is Sahebi & Sui, SPE-228955-MS (ADIPEC 2025), https://doi.org/10.2118/228955-MS.

## Pipeline

Each kind of PDF content takes its own route:

| Content | Route | Stored source |
|---|---|---|
| Fixed-layout tables (casing, mud, bits), activity codes, Non-Prod marks | Word-position readers (code, no model) | `table` / `code` / `column` |
| Other tables, time log | Qwen3-8B, one schema per table | `llm` |
| Free text | Qwen3-8B: event, NPT and activity class | `llm` |
| Vector charts (Pason drilling-parameter graph) | Vector-path and text-layer parser | `reading` / `at_max` / `at_min` |
| Raster figures (End-of-Well Report) | Qwen2.5-VL-7B: transcribe, describe, structure | `llm`, values unverified |

A rule-based trust layer then gives each value a verdict (`pass`, `fail`, `conflict`, `pdf_read`, `weak`, `unchecked`) without an LLM. All results go into a SQLite database (`outputs/drilling_reports.db`, 17 tables). A tool-using agent (Qwen3-8B with `run_sql`, `search_figures` and `get_figure`) answers questions from it.

An interactive map of the pipeline is in `docs/pipeline_map.html`.

## Repository layout

```
prototype/            pipeline, evaluation and agent code (see prototype/README.md)
data/download_forge.py  downloads the public source PDFs and sensor files (not stored here)
data/processed/       extracted reports (JSON) and figure readings, per document D01-D10
data/processed/sensor/  chart channels from the Pason 10 s sensor file of well 78B-32
data/labels/          hand labels by the drilling engineer
data/questions/       the 52 agent questions with gold answers, hand grades and grading notes
outputs/              database, CSV export, chart, NPT and trust evaluations, agent runs and scores
docs/                 pipeline map, grading rules, activity labelling rules
```

## Reproducing the results

Requirements: Python 3.11+, `pip install pymupdf pydantic requests`, and Ollama.

```
python data/download_forge.py            # source PDFs (~60 MB); add --sensors for the Pason CSVs (~2.8 GB)
ollama pull qwen3:8b && ollama pull qwen2.5vl:7b
cd prototype
python run_prototype.py                  # extraction; resumable (--phase text|figures, --only D05)
python trust.py                          # trust verdicts (needs the PDFs; sensor check needs --sensors)
python chart_reader.py                   # vector-chart readings (needs the PDFs)
python build_db.py                       # SQLite database and CSV export
python score_labels.py                   # field accuracy against the hand labels
python eval_charts.py                    # chart readings against the sensor records
python eval_npt.py                       # NPT flag against the operator's marks
python grade_questions.py && python score_ci.py   # agent scores and bootstrap intervals
```

All extracted outputs are already in the repository, so the scoring scripts can be run without re-running the models. Model output depends on the Ollama version; runs used temperature 0 and a fixed seed.

**Evaluation rules.** Reports D01, D02 and D05 were used to tune prompts and are excluded from every reported accuracy. No numbers or text from the evaluated reports appear in the prompts. The agent results in `outputs/agent_questions/` are frozen; a changed agent needs a new held-out question set.

## Data sources and licences

The source documents and sensor records are published by the Utah FORGE project (US Department of Energy, Energy and Geoscience Institute at the University of Utah) on the Geothermal Data Repository under CC BY 4.0:

- Utah FORGE: Well 16A(78)-32 Drilling Data, https://doi.org/10.15121/1776602
- Utah FORGE: Well 56-32 Drilling Data and Logs, https://doi.org/10.15121/1777170
- Utah FORGE: Well 78B-32 Daily Drilling Reports and Logs, https://doi.org/10.15121/1814488

The data in this repository (`data/`, `outputs/`) are derived from those records and are released under **CC BY 4.0** (`LICENSE-DATA.md`). The code is released under the **MIT licence** (`LICENSE`).

## Authors

Hamed Sahebi and Dan Sui, Department of Energy and Petroleum Engineering, University of Stavanger, Norway.

This work was supported by the Research Council of Norway (Proof-of-Concept programme, grant number 349918, project AI-Drill).
