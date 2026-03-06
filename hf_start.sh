#!/bin/bash
# hf_start.sh — SchemeImpactNet HuggingFace Spaces entrypoint
# Runs pipeline (if needed), starts FastAPI on 8000, Streamlit on 7860

set -euo pipefail

echo "============================================================"
echo "  SchemeImpactNet — HuggingFace Spaces Startup"
echo "============================================================"

cd /app

# ── Step 1: Generate / verify processed data ─────────────────────────────────
echo ""
echo "→ Checking processed data..."

NEEDS_PIPELINE=false
for f in data/processed/mnrega_cleaned.csv \
  data/processed/mnrega_predictions.csv \
  data/processed/optimized_budget_allocation.csv; do
  if [[ ! -f "$f" ]]; then
    echo "  Missing: $f"
    NEEDS_PIPELINE=true
  fi
done

if [[ "$NEEDS_PIPELINE" == true ]]; then
  echo "→ Running data pipeline (Stage 3)..."
  python main.py --stage 3
  echo "✓ Pipeline complete"
else
  echo "✓ Processed data found — skipping pipeline"
fi

# ── Step 2: Start FastAPI backend on port 8000 (background) ──────────────────
echo ""
echo "→ Starting FastAPI backend on port 8000..."
python -m uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level warning &
BACKEND_PID=$!

# Wait for backend health
MAX_WAIT=20
WAITED=0
until curl -sf "http://localhost:8000/health" >/dev/null 2>&1; do
  sleep 1
  WAITED=$((WAITED + 1))
  if [[ $WAITED -ge $MAX_WAIT ]]; then
    echo "  ⚠ Backend health timeout — continuing"
    break
  fi
done
echo "✓ Backend live"

# ── Step 3: Start Streamlit on HF port 7860 (foreground) ────────────────────
echo ""
echo "→ Starting Streamlit frontend on port 7860..."
echo "✓ Dashboard: https://huggingface.co/spaces/{YOUR_SPACE}"
echo ""

exec python -m streamlit run frontend/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
