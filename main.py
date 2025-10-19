
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
import math
import ta
from typing import Optional

app = FastAPI(title="Pro Trader AI v3",
              description="Multi-timeframe pro strategy + scalping signals (read-only).",
              version="3.0")

# ---------- helper utilities ----------
def fetch_ohlc_binance(symbol: str, interval: str, limit: int = 500):
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": symbol.upper(), "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume","close_time","qav","num_trades","tb_base","tb_quote","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open_time","open","high","low","close","volume"]]

def ema(series, length):
    return ta.trend.EMAIndicator(series, window=length).ema_indicator()

def rsi(series, length=14):
    return ta.momentum.RSIIndicator(series, window=length).rsi()

def atr(df, length=14):
    return ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=length).average_true_range()

def recent_s_r(df, lookback=50):
    recent_h = df["high"].tail(lookback).max()
    recent_l = df["low"].tail(lookback).min()
    return float(recent_h), float(recent_l)

def score_confidence(*components):
    vals = [max(0.0, min(1.0, float(v))) for v in components if v is not None]
    if not vals:
        return 0.0
    return float(sum(vals)/len(vals))

def fmt(x):
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except:
        return str(x)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:,.2f}"
    return f"{x:.6f}"

# ---------- Models ----------
class SignalOut(BaseModel):
    pair: str
    timeframe_main: str
    timeframe_entry: str
    bias: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    confidence: float
    reason: str

# ---------- PRO multi-timeframe endpoint (swing/position) ----------
@app.get("/pro_signal", response_model=SignalOut)
def pro_signal(pair: str = Query(..., example="BTCUSDT"),
               tf_main: str = Query("1h"),
               tf_entry: str = Query("15m"),
               lookback:int = Query(200)):
    """
    Multi-timeframe pro signal. tf_main: trend timeframe (1h/4h), tf_entry: confirmation timeframe (15m/5m).
    """
    try:
        df_main = fetch_ohlc_binance(pair, tf_main, limit=lookback)
        df_entry = fetch_ohlc_binance(pair, tf_entry, limit=max(200, int(lookback)))
    except Exception as e:
        raise RuntimeError(f"Error fetching data: {e}")

    # prepare indicators
    df_main["close"] = df_main["close"]
    df_entry["close"] = df_entry["close"]
    df_main["ema20"] = ema(df_main["close"], 20)
    df_main["ema50"] = ema(df_main["close"], 50)
    df_main["rsi14"] = rsi(df_main["close"], 14)
    df_entry["ema8"] = ema(df_entry["close"], 8)
    df_entry["ema21"] = ema(df_entry["close"], 21)
    df_entry["rsi14"] = rsi(df_entry["close"], 14)

    # trend alignment
    ma_main_short = df_main["ema20"].iloc[-1] if not df_main["ema20"].isnull().all() else None
    ma_main_long = df_main["ema50"].iloc[-1] if not df_main["ema50"].isnull().all() else None

    trend = "neutral"
    if ma_main_short is not None and ma_main_long is not None:
        trend = "bullish" if ma_main_short > ma_main_long else "bearish"

    # S/R and ATR
    recent_res, recent_sup = recent_s_r(df_main, lookback=60)
    atr_main = atr(df_main, length=14).iloc[-1] if not df_main.empty else None

    # Entry candidate from entry timeframe
    last_entry_close = float(df_entry["close"].iloc[-1])
    if trend == "bullish":
        candidate_entry = max(recent_sup, float(df_entry["ema21"].iloc[-1]))
        sl = recent_sup - (float(atr_main) * 0.5 if atr_main is not None else (recent_sup * 0.01))
        rr = (candidate_entry - sl) if candidate_entry > sl else max(candidate_entry*0.01, 0.0001)
        tp1 = candidate_entry + rr * 1.5
        tp2 = candidate_entry + rr * 2.5
        bias = "long"
        c_trend = 1.0
        c_rsi = 1.0 if 35 < float(df_entry["rsi14"].iloc[-1]) < 70 else 0.6
        c_ma = 1.0 if float(df_entry["ema8"].iloc[-1]) > float(df_entry["ema21"].iloc[-1]) else 0.5
        confidence = score_confidence(c_trend, c_rsi, c_ma)
        reason = f"Trend {trend}. Entry in EMA21 & recent support confluence."
    elif trend == "bearish":
        candidate_entry = min(recent_res, float(df_entry["ema21"].iloc[-1]))
        sl = recent_res + (float(atr_main) * 0.5 if atr_main is not None else (recent_res * 0.01))
        rr = (sl - candidate_entry) if sl > candidate_entry else max(candidate_entry*0.01, 0.0001)
        tp1 = candidate_entry - rr * 1.5
        tp2 = candidate_entry - rr * 2.5
        bias = "short"
        c_trend = 1.0
        c_rsi = 1.0 if 30 < float(df_entry["rsi14"].iloc[-1]) < 65 else 0.6
        c_ma = 1.0 if float(df_entry["ema8"].iloc[-1]) < float(df_entry["ema21"].iloc[-1]) else 0.5
        confidence = score_confidence(c_trend, c_rsi, c_ma)
        reason = f"Trend {trend}. Entry in EMA21 & recent resistance confluence."
    else:
        candidate_entry = last_entry_close
        sl = last_entry_close * 0.99
        tp1 = last_entry_close * 1.01
        tp2 = last_entry_close * 1.02
        bias = "neutral"
        confidence = 0.2
        reason = "No clear trend on main timeframe."

    return SignalOut(
        pair=pair,
        timeframe_main=tf_main,
        timeframe_entry=tf_entry,
        bias=bias,
        entry=round(candidate_entry, 8),
        sl=round(sl, 8),
        tp1=round(tp1, 8),
        tp2=round(tp2, 8),
        confidence=round(confidence, 3),
        reason=reason
    )

# ---------- SCALP endpoint (very short TFs) ----------
@app.get("/scalp_signal", response_model=SignalOut)
def scalp_signal(pair: str = Query(..., example="BTCUSDT"),
                 tf: str = Query("3m"),
                 lookback: int = Query(200)):
    """
    Scalp mode: uses short timeframe (1m/3m/5m). Filters: EMA alignment (8/21), volume spike, RSI, ATR-based SL.
    """
    try:
        df = fetch_ohlc_binance(pair, tf, limit=lookback)
    except Exception as e:
        raise RuntimeError(f"Error fetching data: {e}")

    if df.shape[0] < 30:
        return SignalOut(pair=pair, timeframe_main=tf, timeframe_entry=tf, bias="invalid", entry=0, sl=0, tp1=0, tp2=0, confidence=0.0, reason="Not enough bars")

    df["close"] = df["close"]
    df["high"] = df["high"]
    df["low"] = df["low"]
    df["volume"] = df["volume"].astype(float)
    df["ema8"] = ema(df["close"], 8)
    df["ema21"] = ema(df["close"], 21)
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)
    last = df.iloc[-1]

    ema_fast = float(last["ema8"])
    ema_slow = float(last["ema21"])
    rsi_val = float(last["rsi14"])
    atr_val = float(last["atr14"]) if not math.isnan(last["atr14"]) else (last["close"] * 0.001)

    vol_mean = df["volume"].tail(40).mean()
    vol_spike = float(last["volume"]) > (vol_mean * 1.8 if vol_mean>0 else False)

    bias = "neutral"
    candidate_entry = float(last["close"])
    sl = None; tp1=None; tp2=None; confidence=0.0; reason=""

    if ema_fast > ema_slow and rsi_val > 40 and vol_spike:
        bias="long"
        candidate_entry = float(last["close"])
        sl = candidate_entry - (atr_val * 0.6)
        tp1 = candidate_entry + (atr_val * 0.8)
        tp2 = candidate_entry + (atr_val * 1.4)
        confidence = score_confidence(1.0, 0.9 if vol_spike else 0.6, 0.9 if rsi_val<70 else 0.6)
        reason = "Scalp long: EMA8>EMA21, volume spike, RSI ok."
    elif ema_fast < ema_slow and rsi_val < 60 and vol_spike:
        bias="short"
        candidate_entry = float(last["close"])
        sl = candidate_entry + (atr_val * 0.6)
        tp1 = candidate_entry - (atr_val * 0.8)
        tp2 = candidate_entry - (atr_val * 1.4)
        confidence = score_confidence(1.0, 0.9 if vol_spike else 0.6, 0.9 if rsi_val>30 else 0.6)
        reason = "Scalp short: EMA8<EMA21, volume spike, RSI ok."
    else:
        bias="wait"
        candidate_entry = float(last["close"])
        sl = candidate_entry*0.995
        tp1 = candidate_entry*1.003
        tp2 = candidate_entry*1.006
        confidence = 0.2
        reason = "No clean scalp conditions (need EMA alignment + volume spike)."

    return SignalOut(
        pair=pair,
        timeframe_main=tf,
        timeframe_entry=tf,
        bias=bias,
        entry=round(candidate_entry,8),
        sl=round(sl,8) if sl else 0.0,
        tp1=round(tp1,8) if tp1 else 0.0,
        tp2=round(tp2,8) if tp2 else 0.0,
        confidence=round(confidence,3),
        reason=reason
    )

# Healthcheck
@app.get("/")
def root():
    return {"message":"Pro Trader AI v3 - OK"}
