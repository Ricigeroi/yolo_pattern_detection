import mplfinance as mpf
import pandas as pd
from binance.client import Client
import os
import time

api_key = ""
api_secret = ""

client = Client(api_key, api_secret)

# Настройки
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_15MINUTE
days = 60  # Количество дней (2 месяца)
candles_per_day = 96  # 15 минут * 96 = 24 часа

# Создаем папку для сохранения графиков
save_folder = "btc_charts"
os.makedirs(save_folder, exist_ok=True)

# Начальная дата (60 дней назад)
start_time = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp() * 1000)

# Список всех свечей
all_klines = []

# Запрашиваем данные по 1000 свечей за раз
while len(all_klines) < days * candles_per_day:
    print(f"Запрашиваю данные с {pd.to_datetime(start_time, unit='ms')}...")

    klines = client.get_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_time,
        limit=1000
    )

    if not klines:
        break  # Если данных больше нет, выходим

    all_klines.extend(klines)

    # Обновляем `start_time` на конец последней свечи
    start_time = klines[-1][0] + 1

    # Даем API передышку, чтобы избежать лимитов
    time.sleep(0.2)

# Преобразуем в DataFrame
df = pd.DataFrame(all_klines, columns=[
    "time", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"
])

# Преобразуем `time` в datetime и цены в float
df["time"] = pd.to_datetime(df["time"], unit="ms")
numeric_cols = ["open", "high", "low", "close", "volume"]
df[numeric_cols] = df[numeric_cols].astype(float)

# Устанавливаем индекс
df.set_index("time", inplace=True)

# Разбиваем данные на дни и сохраняем графики
for day in df.resample('D'):
    df_day = day[1]

    if len(df_day) < candles_per_day:
        continue  # Пропускаем неполные дни

    date_str = df_day.index[0].strftime("%Y-%m-%d")
    filename = os.path.join(save_folder, f"btc_chart_{date_str}.png")

    mpf.plot(df_day, type="candle", style="charles", savefig=filename, figsize=(6.4, 6.4))
    print(f"Сохранен график: {filename}")

print("Все графики за 2 месяца сохранены!")
