import matplotlib
matplotlib.use("Agg")
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set_theme(style="whitegrid")
df = pd.read_csv("data/sales_clean.csv", parse_dates=["order_date"])

# KPIs
print("Total Sales :", round(df.sales.sum(), 2))
print("Total Profit:", round(df.profit.sum(), 2))
print("Orders      :", df.order_id.nunique())

# 1. Monthly sales trend
monthly = df.set_index("order_date").resample("MS")["sales"].sum()
monthly.plot(figsize=(10, 4), marker="o", title="Monthly Sales Trend")
plt.savefig("outputs/monthly_trend.png", bbox_inches="tight"); plt.show()

# 2. Sales by category
df.groupby("category")["sales"].sum().plot(kind="bar", title="Sales by Category", figsize=(6, 4))
plt.savefig("outputs/category_sales.png", bbox_inches="tight"); plt.show()

# 3. Top 5 products
df.groupby("product")["sales"].sum().nlargest(5).plot(kind="barh", title="Top 5 Products")
plt.savefig("outputs/top_products.png", bbox_inches="tight"); plt.show()

# 4. Region share
df.groupby("region")["sales"].sum().plot(kind="pie", autopct="%1.1f%%", title="Sales by Region", figsize=(5, 5))
plt.ylabel(""); plt.savefig("outputs/region_pie.png", bbox_inches="tight"); plt.show()

# 5. Heatmap: month vs year
pivot = df.pivot_table(values="sales", index="month", columns="year", aggfunc="sum")
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Sales Heatmap (Month x Year)")
plt.savefig("outputs/heatmap.png", bbox_inches="tight"); plt.show()