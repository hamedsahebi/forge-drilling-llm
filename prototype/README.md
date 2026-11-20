# Drilling-report extraction prototype (public data, local LLMs)

Prototype for the follow-up to SPE-228955-MS: LLMs and an AI agent turn unstructured, public drilling
documents into a queryable database, including information held in figures and plots.

## Data
Utah FORGE (US DOE, CC-BY 4.0) via the Geothermal Data Repository, downloaded to `data/raw/forge/`.

| ID  | Well       | Document                               | Why it is in the set                    |
|-----|------------|----------------------------------------|-----------------------------------------|
| D01–D03 | 16A(78)-32 | Daily drilling/completion reports (WellEz-style form) | Template A, drilling + completion |
| D04–D06 | 56-32  | Daily drilling reports (RIMBase)       | Template B, multi-page                  |
| D07 | 78B-32     | 93-page DDR compilation (35 reports)   | Report splitting at scale               |
| D08 | 56-32      | End-of-Well Report, 86 pages           | Narrative + 54 figures (BHA, MSE, maps) |
| D09 | 78B-32     | Pason drilling-parameter chart (sample pages) | Vector plots with a text layer     |
| D10 | 78B-32     | Mud log (sample pages)                 | Composite log graphics                  |

## Pipeline
1. `parse_pdf.py` – PyMuPDF: layout-preserving text, report splitting, figure detection
   (raster images + vector clusters, logos and ruled tables filtered), captions, text-layer hints.
2. `run_prototype.py` – phase 1: `qwen3:8b` fills `schema.DailyReport` / `WellSummary` (schema-constrained
   JSON, pydantic validation, retry on error). Phase 2: `qwen2.5vl:7b` fills `schema.FigureInfo`
   from image + caption + text-layer words.
3. `build_db.py` – everything into `outputs/drilling_reports.db` + `outputs/csv/`.
4. `agent.py` – tool-calling agent (SQL + figure search) answering cross-report questions.

All models run locally through Ollama (8 GB GPU); no data leaves the machine.

```
python data/download_forge.py      # raw PDFs are not in the repo (CC BY 4.0, ~60 MB)
ollama pull qwen3:8b && ollama pull qwen2.5vl:7b
python run_prototype.py            # resumable; --phase text|figures, --only D05
python build_db.py
python agent.py --demo
```
