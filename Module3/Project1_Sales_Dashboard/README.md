# Project 1 - Sales Dashboard

A complete Business-Intelligence pipeline: generate sales data, analyze it with
Pandas `groupby`, and build a **4-chart dashboard** with Matplotlib.

Pipeline: `raw sales data -> clean -> group & aggregate -> 4 charts on one image -> KPI report`.

The four charts:
- **Revenue by Region** (bar chart)
- **Monthly Trend** (line chart)
- **Revenue by Product** (horizontal bar chart)
- **Category Share** (pie chart)

## Files

| File | What it does |
|---|---|
| `sales_dashboard.py` | Generates 400 synthetic orders, runs the groupby KPI analysis, builds the 2x2 dashboard, saves it as a PNG. |
| `sales_dashboard.png` | The saved 2x2 dashboard image (created when the script runs). |

## How to run

```bash
python sales_dashboard.py
```

Requires `numpy`, `pandas`, `matplotlib` (`pip install numpy pandas matplotlib`).
The script uses the Matplotlib `Agg` backend, so it needs **no GUI** - it just
prints the KPI report and saves `sales_dashboard.png` in the same folder.
Open that PNG afterward to view the dashboard.

## Challenges

1. Add a fifth chart (e.g., a **box plot of revenue by region**) by growing the
   grid to `2x3` (or `1x5`).
2. Annotate the bar values on top of each bar with `plt.text(...)` (section 8.7).
3. Extend the analysis with a `Top 5 orders` table printed in the console.
