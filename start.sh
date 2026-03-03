#!/bin/bash
# ============================================================
#  SchemeImpactNet — Start Script
#  Usage: ./start.sh [options]
#
#  Options:
#    --skip-pipeline   Skip data generation even if files missing
#    --backend-only    Start only the FastAPI backend
#    --frontend-only   Start only the Streamlit frontend
#    --port-backend N  Backend port (default: 8000)
#    --port-frontend N Frontend port (default: 8501)
#    --stage N         Pipeline stage to run if needed (1|2|3, default: 3)
# ============================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKEND_PORT=8000
FRONTEND_PORT=8501
PIPELINE_STAGE=3
SKIP_PIPELINE=false
BACKEND_ONLY=false
FRONTEND_ONLY=false
BACKEND_PID=""
FRONTEND_PID=""

# ── Always resolve project root (where this script lives) ─────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
AMBER='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'
ok() { echo -e "${GREEN}  ✓${RESET}  $*"; }
info() { echo -e "${BLUE}  →${RESET}  $*"; }
warn() { echo -e "${AMBER}  ⚠${RESET}  $*"; }
err() { echo -e "${RED}  ✗${RESET}  $*"; }
hr() { echo -e "${BOLD}──────────────────────────────────────────────────${RESET}"; }

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
  --skip-pipeline) SKIP_PIPELINE=true ;;
  --backend-only) BACKEND_ONLY=true ;;
  --frontend-only) FRONTEND_ONLY=true ;;
  --port-backend)
    BACKEND_PORT="$2"
    shift
    ;;
  --port-frontend)
    FRONTEND_PORT="$2"
    shift
    ;;
  --stage)
    PIPELINE_STAGE="$2"
    shift
    ;;
  *) warn "Unknown option: $1" ;;
  esac
  shift
done

# ── Cleanup handler ───────────────────────────────────────────────────────────
cleanup() {
  echo ""
  hr
  info "Shutting down services…"
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null && ok "Backend stopped"
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null && ok "Frontend stopped"
  hr
}
trap cleanup EXIT INT TERM

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  ◈  SchemeImpactNet — Service Manager${RESET}"
hr
echo ""

# ── Prerequisite checks ───────────────────────────────────────────────────────
info "Checking prerequisites…"

if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
  err "Python not found. Install Python 3.9+."
  exit 1
fi
PYTHON=$(command -v python3 2>/dev/null || command -v python)
ok "Python → $($PYTHON --version 2>&1)"

if ! $PYTHON -m uvicorn --version &>/dev/null; then
  warn "uvicorn not found — attempting install…"
  $PYTHON -m pip install "uvicorn[standard]" --quiet || {
    err "uvicorn install failed."
    exit 1
  }
fi
ok "uvicorn ready"

if ! $PYTHON -m streamlit --version &>/dev/null; then
  warn "streamlit not found — attempting install…"
  $PYTHON -m pip install streamlit --quiet || {
    err "streamlit install failed."
    exit 1
  }
fi
STREAMLIT_VER=$($PYTHON -m streamlit --version 2>&1 | awk '{print $3}')
ok "streamlit $STREAMLIT_VER ready"

STREAMLIT_MAJOR=$(echo "$STREAMLIT_VER" | cut -d. -f1)
STREAMLIT_MINOR=$(echo "$STREAMLIT_VER" | cut -d. -f2)
if [[ "$STREAMLIT_MAJOR" -lt 1 ]] || { [[ "$STREAMLIT_MAJOR" -eq 1 ]] && [[ "$STREAMLIT_MINOR" -lt 36 ]]; }; then
  warn "Streamlit $STREAMLIT_VER — upgrade to 1.36+ for st.navigation():"
  warn "  pip install --upgrade streamlit"
fi

if [[ ! -f "$PROJECT_ROOT/frontend/app.py" ]]; then
  err "frontend/app.py not found at $PROJECT_ROOT/frontend/app.py"
  exit 1
fi
ok "frontend/app.py found"

if [[ ! -f "$PROJECT_ROOT/backend/main.py" ]]; then
  err "backend/main.py not found at $PROJECT_ROOT/backend/main.py"
  exit 1
fi
ok "backend/main.py found"

echo ""

# ── Data pipeline ─────────────────────────────────────────────────────────────
if [[ "$FRONTEND_ONLY" == false && "$SKIP_PIPELINE" == false ]]; then
  PROCESSED_FILES=(
    "$PROJECT_ROOT/data/processed/mnrega_cleaned.csv"
    "$PROJECT_ROOT/data/processed/mnrega_predictions.csv"
    "$PROJECT_ROOT/data/processed/optimized_budget_allocation.csv"
  )

  MISSING=false
  for f in "${PROCESSED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
      warn "Missing: $f"
      MISSING=true
    fi
  done

  if [[ "$MISSING" == true ]]; then
    hr
    info "Processed data not found — running Stage $PIPELINE_STAGE pipeline…"
    info "This may take several minutes on first run."
    hr
    echo ""
    cd "$PROJECT_ROOT" && $PYTHON main.py --stage "$PIPELINE_STAGE" || {
      err "Pipeline failed. Check errors above."
      exit 1
    }
    echo ""
    ok "Pipeline complete"
    hr
    echo ""
  else
    ok "Processed data found — skipping pipeline"
    for f in "${PROCESSED_FILES[@]}"; do
      info "  $(basename $f) ($(wc -l <"$f") rows)"
    done
    echo ""
  fi
fi

# ── Start backend ─────────────────────────────────────────────────────────────
if [[ "$FRONTEND_ONLY" == false ]]; then
  if lsof -i ":$BACKEND_PORT" &>/dev/null 2>&1; then
    warn "Port $BACKEND_PORT already in use — stopping existing process…"
    lsof -ti ":$BACKEND_PORT" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi

  info "Starting FastAPI backend on port $BACKEND_PORT…"
  # Backend must run from project root so 'backend.main' import resolves
  (cd "$PROJECT_ROOT" && $PYTHON -m uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    --reload \
    --log-level warning \
    2>&1 | sed "s/^/  [backend] /") &
  BACKEND_PID=$!

  info "Waiting for backend health check…"
  MAX_WAIT=15
  WAITED=0
  until curl -sf "http://localhost:$BACKEND_PORT/health" &>/dev/null; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
      warn "Backend health check timed out after ${MAX_WAIT}s — continuing anyway"
      break
    fi
  done
  curl -sf "http://localhost:$BACKEND_PORT/health" &>/dev/null && ok "Backend live → http://localhost:$BACKEND_PORT"
  echo ""
fi

# ── Start frontend ────────────────────────────────────────────────────────────
if [[ "$BACKEND_ONLY" == false ]]; then
  if lsof -i ":$FRONTEND_PORT" &>/dev/null 2>&1; then
    warn "Port $FRONTEND_PORT already in use — stopping existing process…"
    lsof -ti ":$FRONTEND_PORT" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi

  info "Starting Streamlit frontend on port $FRONTEND_PORT…"
  cd "$PROJECT_ROOT/frontend"
  $PYTHON -m streamlit run app.py --server.port "$FRONTEND_PORT" --server.headless true --browser.gatherUsageStats false &
  FRONTEND_PID=$!
  cd "$PROJECT_ROOT"

  sleep 2
  ok "Frontend live → http://localhost:$FRONTEND_PORT"
  echo ""
fi

# ── Ready banner ──────────────────────────────────────────────────────────────
hr
echo ""
echo -e "${BOLD}  ◈  SchemeImpactNet is running${RESET}"
echo ""
[[ "$FRONTEND_ONLY" == false ]] && echo -e "  ${GREEN}Backend${RESET}   http://localhost:$BACKEND_PORT"
[[ "$FRONTEND_ONLY" == false ]] && echo -e "  ${GREEN}API docs${RESET}  http://localhost:$BACKEND_PORT/docs"
[[ "$BACKEND_ONLY" == false ]] && echo -e "  ${GREEN}Dashboard${RESET} http://localhost:$FRONTEND_PORT"
echo ""
echo -e "  ${BOLD}Press Ctrl+C to stop all services${RESET}"
echo ""
hr
echo ""

# ── Keep alive ────────────────────────────────────────────────────────────────
wait
