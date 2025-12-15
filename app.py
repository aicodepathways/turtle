from typing import Any, List, Dict
from datetime import datetime
import math

import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, Body, HTTPException

app = FastAPI(title="Turtle + TTM Squeeze API")


# ------------- Helpers -----------------

def normalize(payload: Any) -> List[str]:
    """
    Accepts:
      {"tickers": [...]}
      [{"Ticker":[...]}]
      ["CL=F","ES=F"]
    Returns list of unique, uppercased symbols.
    """
    if isinstance(payload, dict) and "tickers" in payload:
        seq = payload["tickers"]
    elif (
        isinstance(payload, list)
        and len(payload) == 1
        and isinstance(payload[0], dict)
        and "Ticker" in payload[0]
    ):
        seq = payload[0]["Ticker"]
    elif isinstance(payload, list):
        seq = payload
    else:
        raise HTTPException(
            status_code=400,
            detail="Send {'tickers':[...]} or [{'Ticker':[...]}] or a list of tickers.",
        )

    out: List[str] = []
    for t in seq:
        if isinstance(t, str) and t.strip():
            sym = t.strip().upper()
            if sym not in out:
                out.append(sym)
    return out


def fetch_daily(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Simple daily OHLCV fetch via yfinance.
    """
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise HTTPException(status_code=502, detail=f"No data for {symbol}")
    # yfinance sometimes returns multi-index columns; flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


# ------------- TTM Squeeze -------------


def compute_ttm_squeeze(
    df: pd.DataFrame,
    bb_length: int = 20,
    kc_length: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
    mom_length: int = 12,
) -> pd.DataFrame:
    """
    Adds TTM Squeeze-style columns to df (mutates and returns df):

      bb_mid, bb_upper, bb_lower
      kc_mid, kc_upper, kc_lower
      squeeze_on (bool)
      squeeze_off (bool)
      squeeze_fire (bool)         # squeeze just released this bar
      squeeze_fire_long (bool)    # fire + positive momentum
      squeeze_fire_short (bool)   # fire + negative momentum
      momentum (float)
      momentum_direction ('up'/'down'/'flat')
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Bollinger Bands
    bb_mid = close.rolling(bb_length).mean()
    bb_std = close.rolling(bb_length).std(ddof=0)
    bb_upper = bb_mid + bb_std * bb_mult
    bb_lower = bb_mid - bb_std * bb_mult

    # True Range & ATR for Keltner
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=kc_length, adjust=False).mean()

    kc_mid = close.ewm(span=kc_length, adjust=False).mean()
    kc_upper = kc_mid + atr * kc_mult
    kc_lower = kc_mid - atr * kc_mult

    # Squeeze conditions
    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    squeeze_off = (bb_upper > kc_upper) | (bb_lower < kc_lower)

    # Momentum: close relative to its rolling mean (simple approximation)
    mom_base = close.rolling(mom_length).mean()
    momentum = close - mom_base

    momentum_direction = pd.Series(index=df.index, dtype="object")
    momentum_direction[momentum > 0] = "up"
    momentum_direction[momentum < 0] = "down"
    momentum_direction[momentum == 0] = "flat"

    # "Squeeze fired" when we were in squeeze and are now out
    squeeze_fire = squeeze_on.shift(1).fillna(False) & (~squeeze_on.fillna(False))
    squeeze_fire_long = squeeze_fire & (momentum > 0)
    squeeze_fire_short = squeeze_fire & (momentum < 0)

    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["kc_mid"] = kc_mid
    df["kc_upper"] = kc_upper
    df["kc_lower"] = kc_lower
    df["squeeze_on"] = squeeze_on
    df["squeeze_off"] = squeeze_off
    df["squeeze_fire"] = squeeze_fire
    df["squeeze_fire_long"] = squeeze_fire_long
    df["squeeze_fire_short"] = squeeze_fire_short
    df["momentum"] = momentum
    df["momentum_direction"] = momentum_direction

    return df


# ------------- Turtle / Donchian 20 -------------


def compute_turtle_20(df: pd.DataFrame, channel_length: int = 20) -> pd.DataFrame:
    """
    Adds Turtle-style 20-day breakout/breakdown columns (mutates and returns df):

      donchian_high_20
      donchian_low_20
      long_breakout_20   # today's High > prior 20-day High
      short_breakdown_20 # today's Low < prior 20-day Low
    """
    high = df["High"]
    low = df["Low"]

    # Donchian channels use prior N bars (exclude today's bar)
    donch_high = high.shift(1).rolling(channel_length).max()
    donch_low = low.shift(1).rolling(channel_length).min()

    long_breakout = high > donch_high
    short_breakdown = low < donch_low

    df["donchian_high_20"] = donch_high
    df["donchian_low_20"] = donch_low
    df["long_breakout_20"] = long_breakout
    df["short_breakdown_20"] = short_breakdown

    return df


# ------------- FastAPI endpoints -------------


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/indicators/ttm_squeeze")
def ttm_squeeze_endpoint(
    payload: Any = Body(...),
    bb_length: int = 20,
    kc_length: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
    mom_length: int = 12,
    last_n: int = 1,
) -> Dict[str, Any]:
    """
    Compute TTM Squeeze-style indicators for each ticker and return the most recent rows.

    Body: {"tickers": ["CL=F","ES=F", ...]}

    Query params control lengths and multipliers.
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")
    if last_n <= 0:
        last_n = 1

    results = []
    for sym in tickers:
        try:
            df = fetch_daily(sym, period="1y", interval="1d")
            df = compute_ttm_squeeze(
                df,
                bb_length=bb_length,
                kc_length=kc_length,
                bb_mult=bb_mult,
                kc_mult=kc_mult,
                mom_length=mom_length,
            )
            tail = df.tail(last_n)
            rows = []
            for idx, row in tail.iterrows():
                rows.append(
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "close": float(row["Close"]),
                        "squeeze_on": bool(row["squeeze_on"])
                        if not pd.isna(row["squeeze_on"])
                        else False,
                        "squeeze_off": bool(row["squeeze_off"])
                        if not pd.isna(row["squeeze_off"])
                        else False,
                        "squeeze_fire": bool(row["squeeze_fire"])
                        if not pd.isna(row["squeeze_fire"])
                        else False,
                        "squeeze_fire_long": bool(row["squeeze_fire_long"])
                        if not pd.isna(row["squeeze_fire_long"])
                        else False,
                        "squeeze_fire_short": bool(row["squeeze_fire_short"])
                        if not pd.isna(row["squeeze_fire_short"])
                        else False,
                        "momentum": None
                        if pd.isna(row["momentum"])
                        else float(row["momentum"]),
                        "momentum_direction": row.get("momentum_direction"),
                    }
                )
            results.append({"ticker": sym, "bars": rows})
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})
    return {"results": results}


@app.post("/signals/turtle20")
def turtle20_endpoint(
    payload: Any = Body(...),
    channel_length: int = 20,
) -> Dict[str, Any]:
    """
    For each ticker, compute 20-day Donchian breakout/breakdown (Turtle-style System 1).
    Returns only the latest bar.
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")

    results = []
    for sym in tickers:
        try:
            df = fetch_daily(sym, period="1y", interval="1d")
            if len(df) < channel_length + 1:
                raise HTTPException(
                    status_code=502,
                    detail=f"Not enough daily data for {sym} (need at least {channel_length+1} bars).",
                )
            df = compute_turtle_20(df, channel_length=channel_length)
            last_ts = df.index[-1]
            row = df.iloc[-1]

            results.append(
                {
                    "ticker": sym,
                    "date": last_ts.strftime("%Y-%m-%d"),
                    "close": float(row["Close"]),
                    "donchian_high_20": None
                    if pd.isna(row["donchian_high_20"])
                    else float(row["donchian_high_20"]),
                    "donchian_low_20": None
                    if pd.isna(row["donchian_low_20"])
                    else float(row["donchian_low_20"]),
                    "long_breakout_20": bool(row["long_breakout_20"])
                    if not pd.isna(row["long_breakout_20"])
                    else False,
                    "short_breakdown_20": bool(row["short_breakdown_20"])
                    if not pd.isna(row["short_breakdown_20"])
                    else False,
                }
            )
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})
    return {"results": results}


@app.post("/signals/combined")
def combined_endpoint(
    payload: Any = Body(...),
    channel_length: int = 20,
    squeeze_lookback: int = 5,
    bb_length: int = 20,
    kc_length: int = 20,
    bb_mult: float = 2.0,
    kc_mult: float = 1.5,
    mom_length: int = 12,
) -> Dict[str, Any]:
    """
    Combined Turtle 20-day breakout + TTM Squeeze logic.

    For each ticker:
      - computes 20-day Donchian breakout/breakdown
      - computes TTM Squeeze and momentum
      - flags if a Squeeze recently fired (within squeeze_lookback bars)
      - emits simple long/short entry signals:

        signal_long_entry  = (today long_breakout_20)   AND (recent squeeze_fire_long)
        signal_short_entry = (today short_breakdown_20) AND (recent squeeze_fire_short)
    """
    tickers = normalize(payload)
    if not tickers:
        raise HTTPException(status_code=400, detail="Empty tickers list.")
    if squeeze_lookback <= 0:
        squeeze_lookback = 5

    results = []
    for sym in tickers:
        try:
            df = fetch_daily(sym, period="1y", interval="1d")
            if len(df) < max(channel_length + 1, bb_length + mom_length + 5):
                raise HTTPException(
                    status_code=502,
                    detail=f"Not enough daily data for {sym} to compute indicators.",
                )

            df = compute_turtle_20(df, channel_length=channel_length)
            df = compute_ttm_squeeze(
                df,
                bb_length=bb_length,
                kc_length=kc_length,
                bb_mult=bb_mult,
                kc_mult=kc_mult,
                mom_length=mom_length,
            )

            # We care about the last bar for today's breakout
            last_ts = df.index[-1]
            last = df.iloc[-1]

            # Look back a few bars for squeeze fires
            recent = df.tail(squeeze_lookback)
            recent_fire_long = bool(recent["squeeze_fire_long"].fillna(False).any())
            recent_fire_short = bool(recent["squeeze_fire_short"].fillna(False).any())

            long_breakout = bool(last["long_breakout_20"]) if not pd.isna(last["long_breakout_20"]) else False
            short_breakdown = bool(last["short_breakdown_20"]) if not pd.isna(last["short_breakdown_20"]) else False

            signal_long_entry = long_breakout and recent_fire_long
            signal_short_entry = short_breakdown and recent_fire_short

            out = {
                "ticker": sym,
                "date": last_ts.strftime("%Y-%m-%d"),
                "close": float(last["Close"]),
                "donchian_high_20": None
                if pd.isna(last["donchian_high_20"])
                else float(last["donchian_high_20"]),
                "donchian_low_20": None
                if pd.isna(last["donchian_low_20"])
                else float(last["donchian_low_20"]),
                "long_breakout_20": long_breakout,
                "short_breakdown_20": short_breakdown,
                "squeeze_on": bool(last["squeeze_on"]) if not pd.isna(last["squeeze_on"]) else False,
                "squeeze_off": bool(last["squeeze_off"]) if not pd.isna(last["squeeze_off"]) else False,
                "squeeze_fire_recent_long": recent_fire_long,
                "squeeze_fire_recent_short": recent_fire_short,
                "momentum": None
                if pd.isna(last["momentum"])
                else float(last["momentum"]),
                "momentum_direction": last.get("momentum_direction"),
                "signal_long_entry": signal_long_entry,
                "signal_short_entry": signal_short_entry,
            }
            results.append(out)
        except HTTPException as e:
            results.append({"ticker": sym, "error": e.detail})

    return {"results": results}
