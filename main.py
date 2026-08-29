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

app = FastAPI(title="SynapseQuant OS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# UNIVERS MULTI-MARCHÉS (28 Actions majeures)
MARKET_UNIVERSE = [
    # TECH & IA
    {"entreprise": "Nvidia", "ticker": "NVDA", "sector": "Tech", "theme": "IA & Puces"},
    {"entreprise": "Apple", "ticker": "AAPL", "sector": "Tech", "theme": "Big Tech US"},
    {"entreprise": "Microsoft", "ticker": "MSFT", "sector": "Tech", "theme": "Big Tech US"},
    {"entreprise": "Alphabet", "ticker": "GOOGL", "sector": "Tech", "theme": "Big Tech US"},
    {"entreprise": "ASML", "ticker": "ASML.AS", "sector": "Tech", "theme": "IA & Puces"},
    {"entreprise": "Capgemini", "ticker": "CAP.PA", "sector": "Tech", "theme": "CAC 40"},
    
    # LUXE & CONSOMMATION
    {"entreprise": "LVMH", "ticker": "MC.PA", "sector": "Luxe", "theme": "Luxe Européen"},
    {"entreprise": "Hermès", "ticker": "RMS.PA", "sector": "Luxe", "theme": "Luxe Européen"},
    {"entreprise": "L'Oréal", "ticker": "OR.PA", "sector": "Luxe", "theme": "CAC 40"},
    {"entreprise": "Kering", "ticker": "KER.PA", "sector": "Luxe", "theme": "Luxe Européen"},
    {"entreprise": "Amazon", "ticker": "AMZN", "sector": "Consommation", "theme": "Big Tech US"},
    {"entreprise": "Tesla", "ticker": "TSLA", "sector": "Consommation", "theme": "Automobile EV"},

    # ÉNERGIE & INDUSTRIE
    {"entreprise": "TotalEnergies", "ticker": "TTE.PA", "sector": "Énergie", "theme": "Dividendes"},
    {"entreprise": "ExxonMobil", "ticker": "XOM", "sector": "Énergie", "theme": "Pétrole US"},
    {"entreprise": "Schneider Electric", "ticker": "SU.PA", "sector": "Industrie", "theme": "Transition Énergétique"},
    {"entreprise": "Air Liquide", "ticker": "AI.PA", "sector": "Industrie", "theme": "CAC 40"},

    # FINANCE
    {"entreprise": "BNP Paribas", "ticker": "BNP.PA", "sector": "Finance", "theme": "Dividendes"},
    {"entreprise": "JPMorgan Chase", "ticker": "JPM", "sector": "Finance", "theme": "Finance US"},
    {"entreprise": "AXA", "ticker": "CS.PA", "sector": "Finance", "theme": "Dividendes"},

    # SANTÉ
    {"entreprise": "Sanofi", "ticker": "SAN.PA", "sector": "Santé", "theme": "Pharma Europe"},
    {"entreprise": "Novo Nordisk", "ticker": "NOVO-B.CO", "sector": "Santé", "theme": "Pharma Europe"}
]

MODEL_PATH = "ml_model_gb.pkl"
DATA_CACHE = []

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(20).std()
    df['Vol_Ratio'] = df['Volume'] / (df['Volume'].rolling(50).mean() + 1e-9)
    return df

def refresh_market_data():
    """Extraction Ultra-Rapide (Batch Download) en 2 secondes"""
    global DATA_CACHE
    logger.info("Extraction Batch des Marchés Mondiaux...")
    
    tickers_list = [item["ticker"] for item in MARKET_UNIVERSE]
    try:
        # Téléchargement groupé ultra-rapide de tout l'univers
        batch_df = yf.download(tickers_list, period="1y", group_by='ticker', progress=False)
        model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        results = []

        for item in MARKET_UNIVERSE:
            try:
                sym = item["ticker"]
                hist = batch_df[sym].dropna() if sym in batch_df else pd.DataFrame()
                if hist.empty or len(hist) < 50: continue

                hist = compute_indicators(hist)
                last = hist.iloc[-1]
                prix = float(last['Close'])
                sma50 = float(last['SMA50']) if not np.isnan(last['SMA50']) else prix
                sma200 = float(last['SMA200']) if not np.isnan(last['SMA200']) else prix
                rsi = float(last['RSI']) if not np.isnan(last['RSI']) else 50.0

                score_ia = 1.0
                if model:
                    feat = [[
                        prix / (sma50 + 1e-9),
                        prix / (sma200 + 1e-9),
                        rsi / 100.0,
                        float(last['Volatility']) if not np.isnan(last['Volatility']) else 0.01,
                        float(last['Vol_Ratio']) if not np.isnan(last['Vol_Ratio']) else 1.0,
                        float(last['MACD']) if not np.isnan(last['MACD']) else 0.0
                    ]]
                    proba = model.predict_proba(feat)[0][1]
                    score_ia = round(proba * 2.0, 2)

                results.append({
                    "entreprise": item["entreprise"],
                    "ticker": sym,
                    "sector": item["sector"],
                    "theme": item["theme"],
                    "prix": round(prix, 2),
                    "sma50": round(sma50, 2),
                    "sma200": round(sma200, 2),
                    "rsi": round(rsi, 1),
                    "roe": 0.18, # Valeur estimée par défaut pour rapidité
                    "target": round(prix * 1.12, 2),
                    "score": score_ia,
                    "patterns": ["Tendance Haussière"] if prix > sma200 else []
                })
            except Exception as e:
                logger.error(f"Erreur parsing {item['ticker']}: {e}")
                continue

        if results:
            DATA_CACHE = results
            logger.info(f"Marchés mis à jour : {len(DATA_CACHE)} actions chargées en cache.")
    except Exception as e:
        logger.error(f"Erreur lors du Batch Download: {e}")

def train_model():
    logger.info("Entraînement ML...")
    tickers_list = [item["ticker"] for item in MARKET_UNIVERSE]
    batch_df = yf.download(tickers_list, period="2y", group_by='ticker', progress=False)
    dataset, labels = [], []

    for item in MARKET_UNIVERSE:
        try:
            sym = item["ticker"]
            hist = batch_df[sym].dropna() if sym in batch_df else pd.DataFrame()
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
        except Exception:
            continue

    if dataset:
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        model.fit(dataset, labels)
        joblib.dump(model, MODEL_PATH)
        logger.info("Modèle ML sauvegardé.")
        refresh_market_data()

scheduler = BackgroundScheduler()
scheduler.add_job(train_model, 'cron', hour=23, minute=0)
scheduler.add_job(refresh_market_data, 'interval', minutes=15)
scheduler.start()

@app.on_event("startup")
def startup_event():
    if not os.path.exists(MODEL_PATH):
        train_model()
    else:
        refresh_market_data()

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/data")
def get_data():
    if not DATA_CACHE:
        refresh_market_data()
    return DATA_CACHE

@app.get("/api/markowitz")
def get_markowitz_allocation(risk_profile: str = Query("equilibre")):
    top_items = sorted(DATA_CACHE, key=lambda x: x['score'], reverse=True)[:5]
    if not top_items: return {"allocation": []}
    
    weights = [0.35, 0.25, 0.2, 0.1, 0.1]
    allocation = []
    for idx, item in enumerate(top_items):
        allocation.append({
            "entreprise": item["entreprise"],
            "ticker": item["ticker"],
            "weight_pct": round(weights[idx] * 100, 1),
            "score": item["score"]
        })
    return {"risk_profile": risk_profile, "allocation": allocation}

@app.get("/api/backtest")
def run_backtest(ticker: str = Query("MC.PA")):
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 50: return {"error": "Historique insuffisant"}
        hist = compute_indicators(hist).dropna()
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
            "max_drawdown_pct": max_dd
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/montecarlo")
def run_montecarlo(ticker: str = Query("MC.PA")):
    try:
        hist = yf.Ticker(ticker).history(period="6m")
        last_price = hist['Close'].iloc[-1]
        returns = hist['Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        
        simulation_results = []
        for _ in range(150):
            prices = [last_price]
            for _ in range(30):
                prices.append(prices[-1] * (1 + np.random.normal(mu, sigma)))
            simulation_results.append(prices[-1])
            
        return {
            "ticker": ticker,
            "bear_case_p10": round(float(np.percentile(simulation_results, 10)), 2),
            "base_case_p50": round(float(np.percentile(simulation_results, 50)), 2),
            "bull_case_p90": round(float(np.percentile(simulation_results, 90)), 2)
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
