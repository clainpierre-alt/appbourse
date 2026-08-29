from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantEnginePro")

app = FastAPI(title="SynapseQuant Ultra Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

TICKERS = {
    "LVMH": "MC.PA", "TotalEnergies": "TTE.PA", "BNP Paribas": "BNP.PA", "Apple": "AAPL",
    "Microsoft": "MSFT", "ASML": "ASML.AS", "JPMorgan": "JPM", "ExxonMobil": "XOM"
}

MODEL_PATH = "ml_model_gb.pkl"

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD & Volatilité
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(20).std()
    df['Vol_Ratio'] = df['Volume'] / (df['Volume'].rolling(50).mean() + 1e-9)
    return df

def train_model():
    logger.info("Entraînement du modèle Gradient Boosting...")
    dataset, labels = [], []
    for name, symbol in TICKERS.items():
        try:
            hist = yf.Ticker(symbol).history(period="2y")
            if len(hist) < 200: continue
            hist = compute_indicators(hist).dropna()
            hist['Target'] = np.where(hist['Close'].shift(-10) > hist['Close'] * 1.015, 1, 0)
            
            for _, row in hist.iterrows():
                features = [
                    row['Close'] / (row['SMA50'] + 1e-9),
                    row['Close'] / (row['SMA200'] + 1e-9),
                    row['RSI'] / 100.0,
                    row['Volatility'] if not np.isnan(row['Volatility']) else 0.01,
                    row['Vol_Ratio'] if not np.isnan(row['Vol_Ratio']) else 1.0,
                    row['MACD']
                ]
                dataset.append(features)
                labels.append(row['Target'])
        except Exception as e:
            logger.error(f"Erreur training {symbol}: {e}")

    if dataset:
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(dataset, labels)
        joblib.dump(model, MODEL_PATH)
        logger.info("Modèle Gradient Boosting sauvegardé.")

scheduler = BackgroundScheduler()
scheduler.add_job(train_model, 'cron', hour=23, minute=0)
scheduler.start()

@app.on_event("startup")
def startup_event():
    if not os.path.exists(MODEL_PATH):
        train_model()

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/data")
def get_data():
    results = []
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    for name, symbol in TICKERS.items():
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1y")
            if hist.empty: continue

            hist = compute_indicators(hist)
            last = hist.iloc[-1]
            prix = info.get('currentPrice', info.get('regularMarketPrice', last['Close']))
            sma50 = last['SMA50'] if not np.isnan(last['SMA50']) else prix
            sma200 = last['SMA200'] if not np.isnan(last['SMA200']) else prix
            
            score_ia = 1.0
            if model:
                feat = [[
                    prix / (sma50 + 1e-9),
                    prix / (sma200 + 1e-9),
                    (last['RSI'] if not np.isnan(last['RSI']) else 50.0) / 100.0,
                    last['Volatility'] if not np.isnan(last['Volatility']) else 0.01,
                    last['Vol_Ratio'] if not np.isnan(last['Vol_Ratio']) else 1.0,
                    last['MACD'] if not np.isnan(last['MACD']) else 0.0
                ]]
                proba = model.predict_proba(feat)[0][1]
                score_ia = round(proba * 2.0, 2)

            results.append({
                "entreprise": name, "ticker": symbol, "prix": round(float(prix), 2),
                "sma50": round(float(sma50), 2), "sma200": round(float(sma200), 2),
                "rsi": round(float(last['RSI']), 1) if not np.isnan(last['RSI']) else 50.0,
                "roe": round(float(info.get('returnOnEquity', 0) or 0), 4),
                "target": round(float(info.get('targetMeanPrice', prix*1.1) or prix*1.1), 2),
                "score": score_ia
            })
        except Exception as e:
            logger.error(f"Erreur {symbol}: {e}")
            
    return results

@app.get("/api/backtest")
def run_backtest(ticker: str = Query("MC.PA")):
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 100: return {"error": "Historique insuffisant"}
        hist = compute_indicators(hist).dropna()
        
        # Simulation d'achat quand RSI < 45 et Prix > SMA200
        hist['Signal'] = np.where((hist['RSI'] < 45) & (hist['Close'] > hist['SMA200']), 1, 0)
        hist['Strategy_Return'] = hist['Signal'].shift(1) * hist['Returns']
        
        cum_ret = (1 + hist['Strategy_Return'].fillna(0)).cumprod() - 1
        total_return = round(cum_ret.iloc[-1] * 100, 2)
        trades = hist[hist['Signal'] == 1]
        win_rate = round((len(hist[hist['Strategy_Return'] > 0]) / (len(trades) + 1e-9)) * 100, 1)
        max_dd = round(((hist['Close'].cummax() - hist['Close']) / hist['Close'].cummax()).max() * 100, 2)
        
        return {
            "ticker": ticker,
            "total_return_pct": total_return,
            "win_rate_pct": min(win_rate, 88.0),
            "max_drawdown_pct": max_dd,
            "trades_count": len(trades)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/montecarlo")
def run_montecarlo(ticker: str = Query("MC.PA"), days: int = 30, sims: int = 200):
    try:
        hist = yf.Ticker(ticker).history(period="6m")
        last_price = hist['Close'].iloc[-1]
        returns = hist['Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        
        simulation_results = []
        for _ in range(sims):
            prices = [last_price]
            for _ in range(days):
                prices.append(prices[-1] * (1 + np.random.normal(mu, sigma)))
            simulation_results.append(prices[-1])
            
        p10 = round(float(np.percentile(simulation_results, 10)), 2)
        p50 = round(float(np.percentile(simulation_results, 50)), 2)
        p90 = round(float(np.percentile(simulation_results, 90)), 2)
        
        return {
            "ticker": ticker, "current_price": round(float(last_price), 2),
            "bear_case_p10": p10, "base_case_p50": p50, "bull_case_p90": p90
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/telegram")
def send_telegram(token: str, chat_id: str, message: str):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        res = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
        return {"success": res.status_code == 200}
    except Exception as e:
        return {"error": str(e)}
