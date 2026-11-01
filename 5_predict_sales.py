import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/sales_clean.csv", parse_dates=["order_date"])
monthly = df.set_index("order_date").resample("MS")["sales"].sum().reset_index()
monthly["t"] = np.arange(len(monthly))

def make_features(d):
    return pd.DataFrame({
        "t": d["t"].values,
        "peak": d["order_date"].dt.month.isin([11, 12]).astype(int).values,
    })

X, y = make_features(monthly), monthly["sales"]
split = int(len(monthly) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)}
results = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    results[name] = p
    print(f"{name}: MAE={mean_absolute_error(y_test, p):.0f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_test, p)):.0f}  R2={r2_score(y_test, p):.2f}")

# Forecast next 6 months with Linear Regression (extrapolates trend + seasonality)
final = LinearRegression().fit(X, y)
future_dates = pd.date_range(monthly.order_date.max() + pd.offsets.MonthBegin(1), periods=6, freq="MS")
future = pd.DataFrame({"order_date": future_dates,
                       "t": np.arange(len(monthly), len(monthly) + 6)})
future["predicted_sales"] = final.predict(make_features(future))
print(future[["order_date", "predicted_sales"]].round(0))
future.to_csv("outputs/forecast.csv", index=False)

# Plot
plt.figure(figsize=(11, 4))
plt.plot(monthly.order_date, monthly.sales, label="Actual")
plt.plot(monthly.order_date[split:], results["Linear Regression"], "--", label="Test prediction")
plt.plot(future.order_date, future.predicted_sales, "r-o", label="Forecast")
plt.legend(); plt.title("Sales Forecast")
plt.savefig("outputs/forecast.png", bbox_inches="tight"); plt.show()