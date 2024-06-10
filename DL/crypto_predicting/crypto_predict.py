from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd

df = pd.read_csv("/content/gdrive/My Drive/crypto_data/LTC-USD.csv", names=["time", "low", "high", "open", "close", "volume"])

print(df.head())

import os

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"

def classify(current, future):
  if float(future) > float(current):
    return 1
  else:
    return 0

main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
for ratio in ratios:
  dataset = f"/content/gdrive/My Drive/crypto_data/{ratio}.csv"
  df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])
  #print(df.head())
  df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
  df.set_index("time", inplace=True)
  df = df[[f"{ratio}_close", f"{ratio}_volume"]]

  if len(main_df) == 0:
    main_df = df
  else:
    main_df = main_df.join(df, how='outer', lsuffix='_left', rsuffix='_right')

main_df["future"] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

main_df["target"] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(10))

