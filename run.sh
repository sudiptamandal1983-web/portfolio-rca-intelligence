#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run.sh — Banking Portfolio RCA Pipeline
#
# Usage:
#   ./run.sh                        # defaults
#   ./run.sh --llm                  # enable LLM narratives
#   ./run.sh --db path/to/db.db     # custom DB path
#   ./run.sh --out reports/out.json # custom report path
#   ./run.sh --dimensions regional vintage_risk
#   ./run.sh --z 2.5 --min-sample 50
# -----------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Config — edit these defaults if needed
# ---------------------------------------------------------------------------
DB_PATH="data/portfolio.db"
OUT_PATH="reports/audit_$(date +%Y%m%d_%H%M%S).json"
PYTHON=${PYTHON:-python3}

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
echo -e "${CYAN}[1/4] Checking dependencies...${NC}"

MISSING=()
for pkg in duckdb pandas; do
    if ! $PYTHON -c "import $pkg" 2>/dev/null; then
        MISSING+=("$pkg")
    fi
done

# Only check anthropic if --llm flag is passed
if [[ " $* " =~ " --llm " ]]; then
    if ! $PYTHON -c "import anthropic" 2>/dev/null; then
        MISSING+=("anthropic")
    fi
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        echo -e "${RED}❌  --llm flag set but ANTHROPIC_API_KEY is not in env.${NC}"
        echo -e "    Export it first:  export ANTHROPIC_API_KEY=sk-ant-..."
        exit 1
    fi
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️   Missing packages: ${MISSING[*]}${NC}"
    echo -e "    Installing..."
    $PYTHON -m pip install "${MISSING[@]}" --quiet
    echo -e "${GREEN}    Installed.${NC}"
else
    echo -e "${GREEN}    All dependencies present.${NC}"
fi

# ---------------------------------------------------------------------------
# Validate DB
# ---------------------------------------------------------------------------
echo -e "${CYAN}[2/4] Validating database...${NC}"

# Allow --db override from CLI args passed through to this script
RESOLVED_DB="$DB_PATH"
for i in "$@"; do
    if [[ $PREV == "--db" ]]; then RESOLVED_DB="$i"; fi
    PREV="$i"
done

if [[ ! -f "$RESOLVED_DB" ]]; then
    echo -e "${RED}❌  Database not found: $RESOLVED_DB${NC}"
    echo "    Pass --db <path> or check your data directory."
    exit 1
fi
echo -e "${GREEN}    Found: $RESOLVED_DB${NC}"

# ---------------------------------------------------------------------------
# Ensure reports directory exists
# ---------------------------------------------------------------------------
echo -e "${CYAN}[3/4] Preparing output directory...${NC}"
mkdir -p reports
echo -e "${GREEN}    reports/ ready.${NC}"

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
echo -e "${CYAN}[4/4] Running pipeline...${NC}\n"

$PYTHON run.py \
    --db  "$DB_PATH" \
    --out "$OUT_PATH" \
    "$@"   # pass all CLI args through (--llm, --dimensions, --z, etc.)

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo -e "\n${GREEN}✅  Pipeline complete.${NC}"
echo -e "    Report → ${CYAN}${OUT_PATH}${NC}\n"
