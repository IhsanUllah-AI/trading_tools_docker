# 🧠 Trading Tool — Flask + Docker + Railway

A fully containerized Python (Flask) trading analysis tool that integrates multiple market indicators like **Fibonacci**, **Elliott Wave**, **Ichimoku**, **Wyckoff**, and **Gann**.  
Built for easy deployment and sharing — run locally with Docker or deploy instantly on [Railway](https://railway.app).

---

## 📂 Project Structure

─ Myproject/
├── app.py ← Flask app entry point
├── fib.py ← Fibonacci indicator logic
├── elliot.py ← Elliott Wave analysis
├── ichimoku.py ← Ichimoku cloud logic
├── wyckoff.py ← Wyckoff strategy
├── gann.py ← Gann angle & pattern calculations
├── utils.py ← Shared helper functions
├── requirements.txt ← Python dependencies
├── Dockerfile ← Docker setup
├── templates/
│ └── index.html ← Frontend page
└── data/
├── active_trade.json
├── trade_history.json
