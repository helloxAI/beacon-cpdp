"""Build the aggregate tables from the raw result CSVs.

Usage: python make_tables.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments.aggregate import write_all

if __name__ == "__main__":
    write_all()
    print("tables written to results/tables/")
