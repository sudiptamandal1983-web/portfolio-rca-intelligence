# Portfolio RCA Intelligence System

A production-grade multi-agent system for automated portfolio anomaly detection, root cause analysis, and stakeholder reporting. Built on 890k Lending Club loan records.

---

## What it does

The system continuously scans a loan portfolio across multiple analytical dimensions, identifies statistical anomalies, enriches findings with co-movement analysis and macroeconomic context, generates causal narratives via LLM, and delivers role-based insights to stakeholders via email and an interactive dashboard.

```
890k loan records (DuckDB)
        ↓
🔍  Hunter Agent        Z-score anomaly detection × 6 dimensions
        ↓
🔗  Correlation Agent   Co-movement probing × 10 metrics
        ↓
🌐  Macro Agent         FRED macroeconomic regime enrichment
        ↓
🧠  RCA Agent           LLM-generated causal narratives (Claude)
        ↓
✅  Evaluation Agent    Automated narrative quality gate
        ↓
📧  Email Agent         Role-based HTML digest via Gmail SMTP
        ↓
📊  Dashboard           Interactive Streamlit visualisation
        ↓
📄  JSON Report         Full audit trail saved to disk
```

---

## Architecture

### Multi-agent design

Agents communicate through typed message contracts rather than calling each other directly. This means any agent can be swapped, retried, or extended without touching the rest of the pipeline.

```
AnomalyReport  →  EnrichedReport  →  InsightBundle  →  DeliveryReceipt
   (Hunter)       (Correlation)         (RCA)            (Email)
```

### Agents

| Agent | Role | Key output |
|---|---|---|
| `HunterAgent` | Z-score anomaly detection across configurable dimensions | Flagged segments with z-scores |
| `CorrelationAgent` | Probes 10 metrics at each anomalous segment to find co-movers | Risk direction classification |
| `MacroAgent` | Joins FRED macro data at origination and performance dates | Regime label + macro narrative |
| `RCAAgent` | Generates causal narratives (template or LLM) | Human-readable insight text |
| `EvaluationAgent` | Scores LLM narratives on 4 dimensions, falls back to template if below threshold | Quality-gated narrative |
| `EmailAgent` | Routes insights to recipient groups, renders HTML digest | Delivered email receipts |
| `Orchestrator` | Coordinates all agents, owns message bus, writes audit report | `PipelineResult` |

### Analytical dimensions

| Dimension | Metric | What it detects |
|---|---|---|
| Regional | Avg DTI by state | Geographic leverage anomalies |
| Vintage risk | Delinquency rate by cohort × grade | Temporal credit quality drift |
| Credit quality | Composite risk score by loan purpose | Purpose-level underwriting gaps |
| Yield analysis | Avg interest rate by employment tenure | Pricing anomalies by tenure |
| Utilisation stress | Avg revolving utilisation by ownership × grade | Credit stress by segment |
| Income verification | Avg interest rate by verification × purpose | Verification pricing disparity |

### Macro enrichment (FRED)

Seven Federal Reserve time series are joined to each anomalous segment at two points in time:

| Series | What it captures |
|---|---|
| `FEDFUNDS` | Fed funds rate — monetary policy environment |
| `UNRATE` | Unemployment rate — borrower stress |
| `CPIAUCSL` | CPI — inflation environment |
| `UMCSENT` | Consumer sentiment — behavioural signal |
| `BAMLH0A0HYM2` | HY credit spread — market risk appetite |
| `T10Y2Y` | Yield curve slope — recession signal |
| `DRCCLACBS` | Credit card delinquency rate — consumer credit stress |

### Evaluation scoring

LLM narratives are scored on four dimensions before delivery:

| Dimension | What it checks |
|---|---|
| Factual grounding | Every claim traceable to the data |
| Causal validity | Hypothesis supported by co-movers |
| Consistency | Internally coherent, no contradictions |
| Tone | Appropriate for risk committee audience |

Narratives scoring below the threshold (default 3.5/5) automatically fall back to deterministic templates.

---

## Sample output

**Regional anomaly — Washington DC**

> DC borrowers (n=5,356) exhibit substantially lower debt-to-income ratios (15.7 vs. portfolio average 19.6, z=-2.89) alongside significantly higher incomes (z=+2.96), better credit grades (z=-2.28), and lower interest rates (z=-2.56), consistent with improving credit quality rather than elevated risk. This positive shift likely reflects DC's concentration of higher-income professional borrowers who maintained stable employment through the 2016 credit stress period, when tighter underwriting standards may have skewed originations toward prime segments in the region. The pattern appears idiosyncratic to DC's demographic and occupational mix rather than macro-driven, as the credit stress regime would typically have elevated DTI ratios across geographies.

**Vintage risk — November 2014 Grade G**

> The November 2014 Grade G vintage exhibits severely elevated delinquency at 54.1% versus a population average of 19.3% (z=+2.72), despite originating during a moderate macro environment with improving fundamentals (5.8% unemployment, declining to 4.9% by performance period). The co-movement pattern shows mixed risk signals — elevated loan amounts (z=+1.51) without consistently deteriorating credit indicators — suggesting this cohort's underperformance is largely idiosyncratic rather than macro-driven. Given the benign economic backdrop, the anomaly points to potential underwriting relaxation or adverse selection within this specific vintage, warranting a detailed review of origination controls in place during Q4 2014 for subprime segments.

---

## Tech stack

- **Data** — DuckDB (890k Lending Club loan records)
- **Statistical detection** — Z-score anomaly detection with configurable thresholds
- **LLM** — Anthropic Claude (claude-sonnet-4-5) for causal narrative generation
- **Macro data** — Federal Reserve FRED API (7 time series, 2007–2020)
- **Email** — Gmail SMTP with HTML digest rendering
- **Dashboard** — Streamlit + Plotly
- **Language** — Python 3.9+

---

## Project structure

```
portfolio-rca/
├── run.py                  # CLI entry point
├── run.sh                  # Bash runner with dependency checks
├── dashboard.py            # Streamlit dashboard
├── agents/
│   ├── __init__.py
│   ├── messages.py         # Typed message contracts
│   ├── hunter.py           # Anomaly detection + dimension registry
│   ├── correlation_agent.py# Co-movement analysis
│   ├── macro_agent.py      # FRED macro enrichment
│   ├── rca_agent.py        # Narrative generation + LLM prompt builder
│   ├── eval_agent.py       # Narrative quality gate
│   ├── email_agent.py      # HTML digest + Gmail delivery
│   └── orchestrator.py     # Pipeline coordinator
├── data/
│   └── portfolio.db        # DuckDB database (not included — see setup)
└── reports/                # Auto-created — timestamped JSON audit trail
```

---

## Setup

### Prerequisites

- Python 3.9+
- [Lending Club 890k dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) from Kaggle

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/portfolio-rca.git
cd portfolio-rca

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install duckdb pandas anthropic streamlit plotly
```

### Load the data

```bash
# Place the Lending Club CSV in data/
# Then load into DuckDB:
python3 -c "
import duckdb
con = duckdb.connect('data/portfolio.db')
con.execute(\"CREATE TABLE loans AS SELECT * FROM read_csv_auto('data/lending_club_890k.csv')\")
con.close()
print('Done')
"
```

### Environment variables

```bash
# Required for LLM narratives
export ANTHROPIC_API_KEY="sk-ant-..."

# Required for email delivery
export GMAIL_USER="you@gmail.com"
export GMAIL_APP_PASSWORD="your16charpassword"   # Gmail App Password
export RISK_TEAM_EMAILS="risk@bank.com,analyst@bank.com"
export PORTFOLIO_MGR_EMAILS="pm@bank.com"

# Optional — for live FRED data (fallback data included for 2010-2020)
export FRED_API_KEY="your_fred_key"
```

### Gmail App Password setup

1. Enable 2-Step Verification on your Google account
2. Go to `myaccount.google.com → Security → App Passwords`
3. Generate a password for "Mail"
4. Use the 16-character password as `GMAIL_APP_PASSWORD`

---

## Usage

```bash
# Basic run (template narratives, no email)
python3 run.py

# With LLM narratives
python3 run.py --llm

# With LLM + quality gate
python3 run.py --llm --eval

# Full pipeline
python3 run.py --llm --eval --email

# Specific dimensions only
python3 run.py --dimensions regional vintage_risk --llm --eval --email

# Stricter anomaly threshold
python3 run.py --z 2.5 --llm --eval --email

# Interactive dashboard
streamlit run dashboard.py
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--db` | `data/portfolio.db` | DuckDB database path |
| `--llm` | off | Use Claude for narrative generation |
| `--eval` | off | Score LLM narratives, fall back to template if below threshold |
| `--eval-threshold` | 3.5 | Minimum score (1–5) to keep LLM narrative |
| `--email` | off | Send email digests via Gmail |
| `--email-dry-run` | off | Render emails without sending |
| `--dimensions` | all | Subset of dimensions to scan |
| `--z` | 2.0 | Z-score threshold for anomaly detection |
| `--min-sample` | 100 | Minimum group size for scan |
| `--no-macro` | off | Disable FRED macro enrichment |

---

## Email digest

Two recipient tiers with different content depth:

**Risk team** — full anomaly detail, co-movers table, raw data, all dimensions  
**Portfolio manager** — executive summary, top insights only, clean prose

---

## Roadmap

- [ ] Multi-algorithm support — IQR, Isolation Forest, Local Outlier Factor
- [ ] Schema-agnostic mode — works on any tabular dataset via config
- [ ] Additional data source connectors — CSV, Postgres, Snowflake
- [ ] Scheduled runs via cron or Airflow
- [ ] Slack notification agent
- [ ] Historical trend comparison across report runs

---

## Data

This project uses the [Lending Club loan dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (890k records) available on Kaggle. The dataset is not included in this repository.

---

## License

MIT
