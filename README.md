# 📦 Superstore Sales Insights Dashboard 

An **interactive Streamlit dashboard** for analyzing **Superstore Sales Data**.  
It provides KPIs, charts, customer segmentation, and discount impact analysis — similar to BI dashboards like Power BI, but fully in Python.  

---

## 🚀 Features
- 📂 **Upload or use demo data** (works with Sample/Global Superstore dataset)  
- 📊 **KPIs**: Sales, Profit, Orders, Avg Order Value, Profit Margin, MoM growth  
- 🎛️ **Filters**: Date range, Region, Segment, Category, Sub-Category  
- 📈 **Visuals**:  
  - Sales & Profit over time  
  - Sales by Category  
  - Profit by Sub-Category (Treemap)  
  - Top 10 Products by Sales  
  - Discount Impact on Profit  
  - Profit by State  
- 👥 **Customer RFM Segmentation** (Gold / Silver / Bronze tiers)  
- 📥 **Export**: Download filtered data & RFM table as CSV  

---

## ⚡ Quickstart
```bash

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
