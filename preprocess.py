"""Entry point for the data pipeline, kept at the repo root for continuity.

The code lives in config/schema.py, src/parsing/ and src/pipeline/; this
script only runs it. Equivalent: python -m src.pipeline.build_unified_dataset
"""

from src.pipeline.build_unified_dataset import main

if __name__ == "__main__":
    main()
