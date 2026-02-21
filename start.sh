#!/bin/bash
# Start SchemeImpactNet V1
# Usage: ./start.sh

echo "============================================"
echo "  SchemeImpactNet V1 — Starting services"
echo "============================================"

# Run pipeline first if processed data doesn't exist
if [ ! -f "data/processed/mnrega_predictions.csv" ]; then
  echo "[start] Running pipeline to generate data..."
  python main.py --stage 3
fi

echo ""
echo "[start] Starting FastAPI backend on port 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

sleep 2

echo "[start] Starting Streamlit frontend on port 8501..."
streamlit run frontend/app.py --server.port 8501 &
FRONTEND_PID=$!

echo ""
echo "============================================"
echo "  ✅ Services running:"
echo "  Backend  → http://localhost:8000"
echo "  API docs → http://localhost:8000/docs"
echo "  Frontend → http://localhost:8501"
echo "  Press Ctrl+C to stop"
echo "============================================"

# Wait and cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
