import streamlit as st
import pandas as pd
import numpy as np
import re
import fitz  # PyMuPDF
import plotly.express as px
from sklearn.linear_model import LinearRegression

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime

st.set_page_config(page_title="FMCG Sales Dashboard", layout="wide")
st.title("📊 FMCG Sales Comparison Dashboard")

# ---------- Parser ----------
def extract_sales_data(text, year):
    pattern = re.compile(
        rf"({year}-\d{{2}}-\d{{2}})\s+(.+?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
        re.MULTILINE
    )
    matches = pattern.findall(text)
    data = []
    for match in matches:
        date, product, units, price, total = match
        try:
            data.append({
                "Date": date,
                "Product": product.strip(),
                f"Units_{year}": int(units),
                f"Unit_Price_{year}": float(price),
                f"Total_Sales_{year}": float(total)
            })
        except:
            continue
    return pd.DataFrame(data)

def merge_sales(df_2024, df_2025):
    df = pd.merge(df_2024, df_2025, on="Product", how="outer").fillna(0)
    df["Units_Diff"] = df["Units_2025"] - df["Units_2024"]
    df["Sales_Diff"] = df["Total_Sales_2025"] - df["Total_Sales_2024"]
    df["Sales_Change_%"] = df.apply(
        lambda x: ((x["Total_Sales_2025"] - x["Total_Sales_2024"]) / x["Total_Sales_2024"] * 100)
        if x["Total_Sales_2024"] else 0,
        axis=1
    )
    return df.sort_values(by="Sales_Diff", ascending=False)

def generate_pdf_report(data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    elements.append(Paragraph("FMCG Sales Comparison Report", styles['Title']))
    elements.append(Paragraph(f"Generated: {now}", styles['Normal']))
    elements.append(Spacer(1, 12))

    total_2024 = data["Total_Sales_2024"].sum()
    total_2025 = data["Total_Sales_2025"].sum()
    growth_pct = ((total_2025 - total_2024) / total_2024) * 100 if total_2024 else 0

    summary = [
        f"🟦 Products Compared: {len(data)}",
        f"📉 Total Sales 2024: R{total_2024:,.2f}",
        f"📈 Total Sales 2025: R{total_2025:,.2f}",
        f"🔁 Overall Growth: {growth_pct:.2f}%",
    ]
    for line in summary:
        elements.append(Paragraph(line, styles['Normal']))
    elements.append(Spacer(1, 12))

    table_data = [["Product", "Sales 2024", "Sales 2025", "Growth %"]]
    top = data.sort_values(by="Sales_Diff", ascending=False).head(20)
    for _, row in top.iterrows():
        table_data.append([
            row["Product"],
            f"R{row['Total_Sales_2024']:,.2f}",
            f"R{row['Total_Sales_2025']:,.2f}",
            f"{row['Sales_Change_%']:.2f}%",
        ])

    table = Table(table_data, colWidths=[200, 80, 80, 60])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------- ML Prediction ----------
def prepare_time_series(df, year):
    """
    Prepare time series data with 'Date', 'Product', 'Units' for the given year.
    """
    units_col = f"Units_{year}"
    df = df[["Date", "Product", units_col]].copy()
    df.rename(columns={units_col: "Units"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def predict_next_month_qty(df_ts):
    """
    Predict next period (next month) quantity per product using linear regression on time.
    Returns dataframe with columns: Product, Predicted_Units
    """
    predictions = []

    for product, group in df_ts.groupby("Product"):
        group = group.sort_values("Date")
        if len(group) < 2:
            # Not enough data to predict, use last known units or 0
            pred_units = group["Units"].iloc[-1] if not group.empty else 0
            predictions.append({"Product": product, "Predicted_Units": max(0, pred_units)})
            continue

        # Convert dates to ordinal for regression
        X = group["Date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        y = group["Units"].values

        model = LinearRegression()
        model.fit(X, y)

        # Predict for next month: last date + ~30 days
        next_date = group["Date"].max() + pd.DateOffset(months=1)
        next_date_ord = np.array([[next_date.toordinal()]])
        pred_units = model.predict(next_date_ord)[0]
        pred_units = max(0, pred_units)  # no negative prediction

        predictions.append({"Product": product, "Predicted_Units": pred_units})

    return pd.DataFrame(predictions)

# ---------- PDF Upload ----------
uploaded_file = st.file_uploader("Upload FMCG Sales PDF", type=["pdf"])

if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        full_text = "\n".join([page.get_text() for page in doc])

    df_2024 = extract_sales_data(full_text, 2024)
    df_2025 = extract_sales_data(full_text, 2025)

    if df_2024.empty or df_2025.empty:
        st.error("❌ Could not detect both 2024 and 2025 data in the PDF. Please upload a valid report.")
    else:
        df_merged = merge_sales(df_2024, df_2025)

        # Sidebar Filters (always visible)
        st.sidebar.header("🔍 Filter")
        keyword = st.sidebar.text_input("Search product")
        min_pct = st.sidebar.slider("Min % Sales Change", -100, 200, 0)

        filtered = df_merged[
            df_merged["Product"].str.contains(keyword, case=False, na=False) &
            (df_merged["Sales_Change_%"] >= min_pct)
        ]

        # Tabs for content
        tab_summary, tab_charts, tab_details, tab_prediction = st.tabs(
            ["Summary", "Charts", "Details", "Prediction"]
        )

        with tab_summary:
            total_2024 = df_merged["Total_Sales_2024"].sum()
            total_2025 = df_merged["Total_Sales_2025"].sum()
            growth_pct = ((total_2025 - total_2024) / total_2024 * 100) if total_2024 else 0

            # KPIs at top
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Sales 2024 (R)", f"{total_2024:,.2f}")
            col2.metric("Total Sales 2025 (R)", f"{total_2025:,.2f}")
            col3.metric("Overall Growth %", f"{growth_pct:.2f}%")
            col4.metric("Products Compared", f"{len(df_merged)}")

            st.subheader("📈 Top 10 Products by Sales Increase")
            st.dataframe(df_merged.head(10), use_container_width=True)

            st.subheader("📉 Top 10 Products with Sales Decrease")
            st.dataframe(df_merged[df_merged["Sales_Diff"] < 0].sort_values(by="Sales_Diff").head(10), use_container_width=True)

        with tab_charts:
            top_n = st.slider("Show Top N Products by 2025 Sales", min_value=5, max_value=50, value=20)

            top_products = df_merged.sort_values(by="Total_Sales_2025", ascending=False).head(top_n)

            bar_fig = px.bar(
                top_products.melt(id_vars="Product", value_vars=["Total_Sales_2024", "Total_Sales_2025"]),
                x="Product", y="value", color="variable", barmode="group",
                labels={"value": "Sales (R)", "variable": "Year"},
                title="Top Product Sales Comparison (2024 vs 2025)"
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            scatter_fig = px.scatter(
                df_merged, x="Total_Sales_2024", y="Total_Sales_2025",
                size=df_merged["Sales_Change_%"].abs(),
                color="Sales_Change_%",
                hover_name="Product",
                title="2025 vs 2024 Sales Scatter (Bubble = % Growth)",
                labels={"Total_Sales_2024": "2024 Sales", "Total_Sales_2025": "2025 Sales"}
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

        with tab_details:
            st.subheader("📊 Filtered Product Table")
            st.dataframe(filtered, use_container_width=True)

            # Downloads inside details tab
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Filtered CSV", csv, "fmcg_comparison_filtered.csv", "text/csv")

            pdf_buffer = generate_pdf_report(df_merged)
            st.download_button(
                label="📄 Download Full PDF Report",
                data=pdf_buffer,
                file_name="fmcg_sales_comparison.pdf",
                mime="application/pdf"
            )

        with tab_prediction:
            st.subheader("🤖 Predicted Quantity Sales for Next Period")

            # Prepare time series combining both years for prediction
            df_ts_2024 = prepare_time_series(df_2024, 2024)
            df_ts_2025 = prepare_time_series(df_2025, 2025)
            df_ts_all = pd.concat([df_ts_2024, df_ts_2025])

            predictions_df = predict_next_month_qty(df_ts_all)

            st.dataframe(predictions_df.sort_values(by="Predicted_Units", ascending=False), use_container_width=True)

else:
    st.info("📎 Please upload a PDF report containing sales data")
