import pandas as pd
import numpy as np
import os

DATA_DIR = "data/train"
OUTPUT_DIR = "data/train"   # overwrite safely

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

for file in files:
    print(f"Processing {file}...")

    df = pd.read_csv(os.path.join(DATA_DIR, file))
    df["Date"] = pd.to_datetime(df["Date"])

    # --- Returns ---
    df["ret_1d"] = df["Close"].pct_change()
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_10d"] = df["Close"].pct_change(10)

    # --- EMAs ---
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()

    df["price_vs_ema20"] = df["Close"] / ema20 - 1
    df["ema20_vs_ema50"] = ema20 / ema50 - 1

    # --- RSI (14) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal

    # --- ATR (14 normalized) ---
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df["atr_norm"] = atr / df["Close"]

    # --- Volatility ---
    df["vol_20d"] = df["ret_1d"].rolling(20).std()

    # --- Volume ratio ---
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # --- OBV ---
    obv = np.where(df["Close"] > df["Close"].shift(), df["Volume"],
                   np.where(df["Close"] < df["Close"].shift(), -df["Volume"], 0))
    df["obv"] = pd.Series(obv).cumsum()

    # --- Agent state placeholders ---
    df["position"] = 0
    df["position_size"] = 0
    df["unrealized_pnl"] = 0
    df["time_in_trade"] = 0

    # Drop rows with NaNs from rolling windows
    df = df.dropna().reset_index(drop=True)

    df.to_csv(os.path.join(OUTPUT_DIR, file), index=False)

    print(f"Saved features → {file}")

print("Asset feature engineering completed.")
