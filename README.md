
# Superstore Sales Insights Dashboard — Built by Shubh Kumar

An interactive Streamlit dashboard for analyzing Superstore sales, profitability, discounts, and customer value.
Includes a Power BI–style theme toggle, auto-generated Insights, and a bundled sample dataset so it runs out-of-the-box.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```
If you don't have a dataset, the app automatically loads the bundled sample at `data/sample_superstore.csv`.

## Features
- KPIs: Sales, Profit, Orders, Average Order Value, Margin, MoM delta
- Filters: Date range, Region, Segment, Category, Sub-Category
- Visuals: Time series, Category bars, Sub-category treemap, Top products, Discount impact, Profit by state
- **Insights**: Auto-written bullet points based on current filters (great for interviews)
- **Theme toggle**: Light (Power BI–style) vs Dark
- Exports: Download filtered data and RFM
- Customer analytics: RFM segmentation (Gold/Silver/Bronze)

## Data Schema
Expected columns (case-insensitive):
```
Order ID, Order Date, Ship Date, Ship Mode, Customer ID, Customer Name,
Segment, Country, City, State, Postal Code, Region, Product ID, Category,
Sub-Category, Product Name, Sales, Quantity, Discount, Profit
```

## Screenshots
> Run locally and take screenshots with your data/filters. Add them here for your portfolio.

---
Built with ❤️ using Streamlit + Plotly.
