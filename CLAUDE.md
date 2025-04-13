# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Commands
- Run Python scripts: `python project/wordCount.py`
- Jupyter notebooks: `jupyter notebook project/report/milestone_report.ipynb`
- Dataset exploration: `python -c "import pandas as pd; print(pd.read_csv('project/dataset/NewsFakeCOVID-19.csv').head())"`
- Update repository: `bash update.sh`

## Style Guidelines
- Import order: standard library, third-party, local modules
- Use PEP 8 conventions for Python code
- Variable naming: snake_case for variables and functions
- Error handling: Use try/except with specific exceptions
- Docstrings: Use descriptive docstrings for functions and classes
- Avoid hardcoded paths; use relative paths when possible
- Maintain consistent 4-space indentation
- For data science code, follow common pandas/numpy patterns
- Comment complex operations or algorithms