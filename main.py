import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

con = sqlite3.connect("db/citybikes.db")


weather_df = pd.read_sql("SELECT * FROM weather ", con)
# 1. Замінюємо аномальні -5.0 на класичний пропуск (NaN)
weather_df['precip_mm'] = weather_df['precip_mm'].replace(-5.0, np.nan)

# 2. Заповнюємо ці 10 дірок лінійною інтерполяцією
weather_df['precip_mm'] = weather_df['precip_mm'].interpolate(method='linear')

#формат дати
weather_df["date"] = pd.to_datetime(weather_df["date"])
weather_df["date"] = weather_df["date"].dt.date

rides_df = pd.read_sql("SELECT * FROM rides ", con)

dup_keys = ["bike_id","user_id","start_station_id","end_station_id","start_time","end_time"]

rides_df = rides_df.drop_duplicates(subset= dup_keys)

rides_df['day_data'] = pd.to_datetime(rides_df['start_time'])
rides_df['day_data'] =  rides_df['day_data'].dt.date

daily_rides = rides_df.groupby("day_data")["ride_id"].count().reset_index(name='trips_count')

final = pd.merge(weather_df , daily_rides, left_on="date", right_on='day_data', how = "inner")
final = final.drop(columns = ['day_data'])
final['date'] = pd.to_datetime(final['date'])
final['month'] = final['date'].dt.month
final['day_of_the_week'] = final['date'].dt.dayofweek


train_df = final[final['date']< '2024-01-01']
test_df = final[final['date'] >= '2024-01-01']

x_train = train_df.drop(columns=["trips_count", "date"])
y_train = train_df["trips_count"]
x_test = test_df.drop(columns=["trips_count", "date"])
y_test = test_df["trips_count"]

model = LinearRegression()  

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(rmse , mae, r2)
print(model.coef_)



model_rf  = RandomForestRegressor(random_state=42 , n_estimators=200, max_depth= 6
                                  ,min_samples_split=5)
model_rf.fit(x_train, y_train)
y_pred_rf = model_rf.predict(x_test)

mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)
print(rmse , mae, r2)
print(model_rf.feature_importances_)


plt.figure(figsize=(12, 5))
plt.plot(final["date"], final["trips_count"])
plt.title("Динаміка поїздок за 2 роки")
plt.xlabel("Дата"); plt.ylabel("Поїздок за день")
plt.grid(linestyle='--',alpha = 0.4)
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(
    final["temp_c"],
    final["trips_count"],
    alpha= 0.4,
    edgecolors = "none",
    s = 20
)
plt.title(
    "Вплив температури повітря на кількість поїздок",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Температура навколишнього середовища (°C)", fontsize=11)
plt.ylabel("Кількість поїздок за день (шт.)", fontsize=11)

plt.grid(linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 5))
plt.title("Порівняння: будні vs вихідні")
plt.xlabel("Дні")
plt.ylabel("Кількість поїздок")
data_bar = final.groupby(final["day_of_the_week"] >= 5)["trips_count"].mean()
plt.bar(["Будні", "Вихідні"], data_bar)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(test_df["date"], y_test, label="Реальність")
plt.plot(test_df["date"], y_pred_rf, label="Прогноз RF")
plt.legend()   
plt.title("Прогноз проти реальності (тест 2024)")
plt.xlabel("Дата"); plt.ylabel("Поїздок за день")
plt.grid(linestyle="--", alpha=0.4)
plt.show()

