import boto3
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np



BUCKET_NAME = "superfresh-bigdata"
FILE_KEY = "superfresh_sales.csv"

s3 = boto3.client("s3")

obj = s3.get_object(Bucket=BUCKET_NAME, Key=FILE_KEY)
data = obj["Body"].read().decode("utf-8")

df = pd.read_csv(StringIO(data))

print("Datos cargados desde S3:")
print(df.head())



df["date"] = pd.to_datetime(df["date"])

df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

features = [
    "store_id",
    "product_id",
    "promotion",
    "temperature",
    "day",
    "month",
    "day_of_week"
]

target = "sales"

X = df[features]
y = df[target]




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nEvaluación del modelo:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")



future_data = pd.DataFrame({
    "store_id": [1, 1, 2, 2],
    "product_id": [101, 102, 101, 102],
    "promotion": [1, 0, 1, 0],
    "temperature": [22, 22, 23, 23],
    "day": [8, 8, 8, 8],
    "month": [1, 1, 1, 1],
    "day_of_week": [0, 0, 0, 0]
})

future_predictions = model.predict(future_data)

future_data["predicted_sales"] = future_predictions.round(0).astype(int)

print("\nPredicciones futuras:")
print(future_data)




future_data.to_csv("predictions.csv", index=False)

print("\nArchivo predictions.csv generado correctamente.")

import matplotlib.pyplot as plt

sales_by_date = df.groupby("date")["sales"].sum()

plt.figure(figsize=(10, 5))
plt.plot(sales_by_date.index, sales_by_date.values)
plt.title("Evolución de ventas por fecha")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ventas_por_fecha.png")
plt.show()

sales_by_product = df.groupby("product_id")["sales"].sum()

plt.figure(figsize=(8, 5))
plt.bar(sales_by_product.index.astype(str), sales_by_product.values)
plt.title("Ventas totales por producto")
plt.xlabel("Producto")
plt.ylabel("Ventas")
plt.tight_layout()
plt.savefig("ventas_por_producto.png")
plt.show()