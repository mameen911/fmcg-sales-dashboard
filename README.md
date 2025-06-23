# FMCG Sales Dashboard

📊 **FMCG Sales Comparison Dashboard** is a Streamlit app that extracts and analyzes FMCG product sales data from PDF reports. It provides interactive visualizations, detailed comparison tables, downloadable reports, and machine learning-based sales quantity predictions.

---

## Features

- Extracts sales data (units, unit price, total sales) by product and date from uploaded PDF reports.
- Compares product sales between June 2024 and June 2025.
- Interactive filters to search products and filter by sales growth percentage.
- Summary KPIs showing total sales, growth, and product counts.
- Visualizations including bar charts and scatter plots for sales comparison.
- Detailed tables with sortable and filterable product data.
- Downloadable CSV and PDF reports of sales comparison.
- ML-powered prediction of next period's sales quantities per product using linear regression.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Recommended: Create and activate a virtual environment

### Install dependencies

```bash
pip install streamlit pandas numpy pymupdf plotly scikit-learn reportlab
````

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/fmcg-sales-dashboard.git
cd fmcg-sales-dashboard
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Upload your FMCG sales PDF report containing sales data for June 2024 and June 2025.

4. Explore the dashboard tabs:

   * **Summary:** View KPIs and top products by sales increase/decrease.
   * **Charts:** Visualize sales comparison via bar and scatter plots.
   * **Details:** Filter and download detailed sales data.
   * **Prediction:** View predicted sales quantities for next period per product.

---

## File Format and Parsing

* The app expects the PDF to contain lines matching the pattern:

  ```
  YYYY-MM-DD ProductName Units UnitPrice TotalSales
  ```

  For years 2024 and 2025 separately.

* The regex extracts dates, product names, units sold, unit prices, and total sales.

---

## Machine Learning Prediction

* Uses linear regression on historical monthly sales units per product.
* Predicts next month’s quantity sales for each product.
* If insufficient data points, uses last known units or zero.

---
