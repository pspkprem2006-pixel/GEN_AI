"""
Project 1 - Sales Dashboard (Module 3: Data Analysis & Visualization)

A complete Business-Intelligence pipeline:
    raw sales data -> clean -> group & aggregate -> 4 charts on one image -> KPI report

The four charts: Revenue by Region (bar), Monthly Trend (line),
Revenue by Product (horizontal bar), Category Share (pie).

The dashboard is saved as 'sales_dashboard.png' (Matplotlib 'Agg' backend,
no GUI needed). Run with:  python sales_dashboard.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_data(seed=42):
    """Create 400 synthetic sales orders (no external CSV needed)."""
    rng = np.random.default_rng(seed)
    n_orders = 400

    regions = ["East", "West", "North", "South"]
    products = ["Laptop", "Phone", "Tablet", "Monitor", "Headphones"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    df = pd.DataFrame({
        "OrderID": [f"ORD{i + 1:04d}" for i in range(n_orders)],
        "Region": rng.choice(regions, n_orders, p=[0.35, 0.25, 0.22, 0.18]),
        "Product": rng.choice(products, n_orders, p=[0.30, 0.24, 0.18, 0.16, 0.12]),
        "Month": pd.Categorical(rng.choice(months, n_orders),
                                categories=months, ordered=True),
        "Revenue": rng.normal(90740, 35000, n_orders).clip(5000).round(),
    })
    return df


def analyze(df):
    """The heart of the dashboard - a few groupby calls (section 5.3)."""
    revenue_by_region = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
    revenue_by_product = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
    revenue_by_month = df.groupby("Month")["Revenue"].sum().sort_index()
    revenue_by_category = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
    return {
        "total_revenue": int(df["Revenue"].sum()),
        "orders": len(df),
        "best_region": revenue_by_region.index[0],
        "best_product": revenue_by_product.index[0],
        "by_region": revenue_by_region,
        "by_month": revenue_by_month,
        "by_product": revenue_by_product,
        "by_category": revenue_by_category,
    }


def make_dashboard(kpis):
    """Build the 2x2 subplot dashboard and save it as a PNG (section 12.3)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    kpis["by_region"].plot(kind="bar", ax=axes[0, 0])   # Pandas can plot directly!
    axes[0, 0].set_title("Revenue by Region")
    axes[0, 0].set_ylabel("Revenue")

    kpis["by_month"].plot(kind="line", ax=axes[0, 1], marker="o")
    axes[0, 1].set_title("Monthly Revenue Trend")
    axes[0, 1].set_ylabel("Revenue")

    kpis["by_product"].sort_values().plot(kind="barh", ax=axes[1, 0])
    axes[1, 0].set_title("Revenue by Product")
    axes[1, 0].set_xlabel("Revenue")

    axes[1, 1].pie(kpis["by_category"].values,
                   labels=list(kpis["by_category"].index),
                   autopct="%1.1f%%")
    axes[1, 1].set_title("Category Share")

    fig.suptitle("Sales Dashboard", fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = "sales_dashboard.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    df = generate_data()
    kpis = analyze(df)

    print(f"Total revenue : {kpis['total_revenue']:,}")
    print(f"Orders        : {kpis['orders']}")
    print(f"Best region   : {kpis['best_region']}")
    print(f"Best product  : {kpis['best_product']}")

    path = make_dashboard(kpis)
    print(f"-> Open '{path}' to view the dashboard.")


if __name__ == "__main__":
    main()
