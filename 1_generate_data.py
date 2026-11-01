import pandas as pd, numpy as np
np.random.seed(42)
n = 6000

all_days = pd.date_range("2021-01-01", "2024-12-31")
w = 1 + 0.6 * all_days.month.isin([11, 12]) + 0.0012 * np.arange(len(all_days))  # seasonality + growth
w = w / w.sum()
dates = np.random.choice(all_days, n, p=w)

products = {
    "Laptop": ("Technology", 800), "Phone": ("Technology", 500),
    "Headphones": ("Technology", 80), "Office Chair": ("Furniture", 150),
    "Desk": ("Furniture", 250), "Bookshelf": ("Furniture", 120),
    "Notebook": ("Office Supplies", 5), "Pen Pack": ("Office Supplies", 8),
    "Printer Paper": ("Office Supplies", 20),
}
prod = np.random.choice(list(products), n)
qty = np.random.randint(1, 6, n)
price = np.array([products[p][1] for p in prod])

df = pd.DataFrame({
    "order_id": np.arange(1000, 1000 + n),
    "order_date": dates,
    "customer_id": np.random.randint(1, 401, n),
    "region": np.random.choice(["North", "South", "East", "West"], n),
    "product": prod,
    "category": [products[p][0] for p in prod],
    "quantity": qty,
    "unit_price": price,
})
df["sales"] = df.quantity * df.unit_price
df["profit"] = (df.sales * np.random.uniform(0.05, 0.3, n)).round(2)
df.to_csv("data/sales.csv", index=False)
print("Saved", df.shape)