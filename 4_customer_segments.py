import pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sales_clean.csv", parse_dates=["order_date"])
snapshot = df.order_date.max() + pd.Timedelta(days=1)

rfm = df.groupby("customer_id").agg(
    Recency=("order_date", lambda x: (snapshot - x.max()).days),
    Frequency=("order_id", "nunique"),
    Monetary=("sales", "sum"),
)

scaled = StandardScaler().fit_transform(rfm)
rfm["Segment"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(scaled)

print(rfm.groupby("Segment").mean().round(1))   # interpret: high Monetary + low Recency = best customers
rfm.plot.scatter(x="Frequency", y="Monetary", c="Segment", cmap="viridis", figsize=(7, 5))
plt.title("Customer Segments (K-Means)")
plt.savefig("outputs/customer_segments.png", bbox_inches="tight"); plt.show()