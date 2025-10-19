# Pro Trader AI v3 (Swing + Scalping)

This project provides two main endpoints:
- /pro_signal  -> multi-timeframe professional signals (swing/position)
- /scalp_signal -> scalping signals (1m/3m/5m)

How to deploy:
1. Upload to GitHub repo
2. Connect repo to Railway (or other hosting)
3. Ensure requirements.txt includes needed packages
4. Deploy and use endpoints, e.g.:
   GET /pro_signal?pair=BTCUSDT&tf_main=1h&tf_entry=15m
   GET /scalp_signal?pair=BTCUSDT&tf=3m

Notes:
- Live data fetched from Binance public API (read-only)
- Test thoroughly and backtest strategies before live trading
