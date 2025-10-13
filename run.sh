#!/bin/bash

source /app/.venv/bin/activate

python scripts/downloadData.py
python src/build.py
python src/evaluate.py
