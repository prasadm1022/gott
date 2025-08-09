from pathlib import Path

from kb.builder import build_index
from kb.materialize import materialize
from pipeline.data_pipeline import run_pipeline

# ---------------------------
# Paths
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EMB = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    print("\nStep 1: Converting raw CSVs to tidy format...")
    run_pipeline(PROJECT_ROOT)

    print("\nStep 2: Materializing data...")
    materialize(PROJECT_ROOT)

    print("\nStep 3: Indexing knowledge-base...")
    build_index(PROJECT_ROOT, DEFAULT_EMB)

    print("\nAll steps completed!")

    return 0


if __name__ == "__main__":
    main()
