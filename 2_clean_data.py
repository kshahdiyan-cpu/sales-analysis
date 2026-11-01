import pandas as pd

df = pd.read_csv("data/sales.csv")
print(df.info())
print("Missing:\n", df.isnull().sum())

df = df.drop_duplicates().dropna()
df["order_date"] = pd.to_datetime(df["order_date"])
df = df[(df.quantity > 0) & (df.sales > 0)]

df["year"] = df.order_date.dt.year
df["month"] = df.order_date.dt.month
df["month_name"] = df.order_date.dt.month_name()

df.to_csv("data/sales_clean.csv", index=False)
print("Clean data saved:", df.shape)