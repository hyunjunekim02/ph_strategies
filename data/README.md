# Data Instructions (Investing.com)

This project expects **daily closing price** CSV files downloaded from **Investing.com**.

## Required format
Each CSV must contain the following columns:

- `Date` (e.g., `2025-12-31`)
- `Price` (daily close)

> The pipeline will parse and clean the `Price` column automatically.

## Where to put files
Place the downloaded CSV files in this directory:
