"""
database.py
-----------
SQLite database setup using SQLAlchemy.
Seeds from processed CSVs on first run.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "schemeimpactnet.db")
DB_URL   = f"sqlite:///{DB_PATH}"

engine       = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base         = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_database() -> None:
    """Load processed CSVs into SQLite tables on startup."""
    processed = os.path.join(BASE_DIR, "data", "processed")

    files = {
        "district_data":   os.path.join(processed, "mnrega_cleaned.csv"),
        "predictions":     os.path.join(processed, "mnrega_predictions.csv"),
        "optimizer":       os.path.join(processed, "optimized_budget_allocation.csv"),
    }

    with engine.connect() as conn:
        for table, path in files.items():
            if not os.path.exists(path):
                print(f"[db] WARNING: {path} not found, skipping")
                continue
            df = pd.read_csv(path)
            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"[db] Seeded '{table}': {len(df)} rows")
        conn.commit()

    print("[db] Database ready ✓")
