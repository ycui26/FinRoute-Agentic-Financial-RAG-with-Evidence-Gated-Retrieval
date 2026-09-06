# Data directory

This repository does not redistribute financial filings or benchmark labels.

Expected local layout:

```text
data/
├── documents.csv
├── evaluation.jsonl       # optional, ignored by Git
└── pdfs/                  # ignored by Git
    ├── filing_1.pdf
    └── filing_2.pdf
```

`documents.csv` must contain:

| Column | Description |
| --- | --- |
| `doc_name` | Stable filing identifier used by retrieval and evaluation |
| `file_name` | PDF filename under `data/pdfs/` |
| `company` | Company name |
| `year` | Filing or fiscal year |
| `form` | Filing type such as `10-K` or `10-Q` |

Gold labels belong only in `evaluation.jsonl`; they are never loaded by the online pipeline.
