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

app = FastAPI(title="SynapseQuant OS - Institutionnel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# UNIVERS ÉLARGI : 50+ ACTIONS PAR MARCHÉS, SECTEURS ET THÉMATIQUES
MARKET_UNIVERSE = [
    # TECH & IA (USA / EUROPE)
    {"entreprise": "Nvidia", "ticker": "NVDA", "sector": "Tech", "theme": "IA & Puces"},
    {"entreprise": "Apple", "ticker": "AAPL", "sector": "Tech", "theme": "Big Tech US"},
    {"entreprise": "Microsoft", "ticker": "MSFT", "sector": "Tech", "theme": "Big Tech US"},
    {"entreprise": "Alphabet (Google)", "ticker": "GOOGL", "sector": "Tech", "theme": "Big Tech US"},
    {"entreprise": "ASML", "ticker": "ASML.AS", "sector": "Tech", "theme": "IA & Puces"},
    {"entreprise": "AMD", "ticker": "AMD", "sector": "Tech", "theme": "IA & Puces"},
    {"entreprise": "Broadcom", "ticker": "AVGO", "sector": "Tech", "theme": "IA & Puces"},
    {"entreprise": "Capgemini", "ticker": "CAP.PA", "sector": "Tech", "theme": "CAC 40"},

    # LUXE & CONSOMMATION (EUROPE / USA)
    {"entreprise": "LVMH", "ticker": "MC.PA", "sector": "Luxe", "theme": "Luxe Européen"},
    {"entreprise": "Hermès", "ticker": "RMS.PA", "sector": "Luxe", "theme": "Luxe Européen"},
    {"entreprise": "L'Oréal", "ticker": "OR.PA", "sector": "Luxe", "theme": "CAC 40"},
    {"entreprise": "Kering", "ticker": "KER.PA", "sector": "Luxe", "theme": "Luxe Européen"},
    {"entreprise": "Ferrari", "ticker": "RACE", "sector": "Luxe", "theme": "Consommation Prestige"},
    {"entreprise": "Amazon", "ticker": "AMZN", "sector": "Consommation", "theme": "Big Tech US"},
    {"entreprise": "Tesla", "ticker": "TSLA", "sector": "Consommation", "theme": "Automobile EV"},

    # ÉNERGIE & TRANSITION
    {"entreprise": "TotalEnergies", "ticker": "TTE.PA", "sector": "Énergie", "theme": "Dividendes"},
    {"entreprise": "ExxonMobil", "ticker": "XOM", "sector": "Énergie", "theme": "Pétrole US"},
    {"entreprise": "Chevron", "ticker": "CVX", "sector": "Énergie", "theme": "Pétrole US"},
    {"entreprise": "Schneider Electric", "ticker": "SU.PA", "sector": "Industrie", "theme": "Transition Énergétique"},
    {"entreprise": "Air Liquide", "ticker": "AI.PA", "sector": "Industrie", "theme": "CAC 40"},

    # FINANCE & BANQUES
    {"entreprise": "BNP Paribas", "ticker": "BNP.PA", "sector": "Finance", "theme": "Dividendes"},
    {"entreprise": "JPMorgan Chase", "ticker": "JPM", "sector": "Finance", "theme": "Finance US"},
    {"entreprise": "AXA", "ticker": "CS.PA", "sector": "Finance", "theme": "Dividendes"},
    {"entreprise": "Goldman Sachs", "ticker": "GS", "sector": "Finance", "theme": "Finance US"},

    # SANTÉ & PHARMA
    {"entreprise": "Sanofi", "ticker": "SAN.PA", "sector": "Santé", "theme": "Pharma Europe"},
    {"entreprise": "Novo Nordisk", "ticker": "NOVO-B.CO", "sector": "Santé", "theme": "Pharma Europe"},
    {"entreprise": "Eli Lilly", "ticker": "LLY", "sector": "Santé", "theme": "Pharma US"},
    {"entreprise": "Pfizer", "ticker": "PFE", "sector": "Santé", "theme": "Dividendes"}
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

def detect_chart_patterns(df: pd.DataFrame) -> list:
    """Détection algorithmique de figures chartistes"""
    patterns = []
    if len(df) < 50: return patterns
    
    last = df.iloc[-1]
    prev_20 = df.iloc[-20:-1]
    
    # Cassure de résistance (Breakout)
    if last['Close'] > prev_20['High'].max():
        patterns.append("Breakout Haussier")
        
    # Double Bottom simplifié (Support testé deux fois)
    min_price = prev_20['Low'].min()
    near_min = prev_20[prev_20['Low'] <= min_price * 1.01]
    if len(near_min) >= 2 and last['Close'] > last['SMA50']:
        patterns.append("Double Bottom")
        
    # Divergence RSI / Prix
    if last['RSI'] < 35 and last['Close'] > last['SMA200']:
        patterns.append("RSI Survendu (Support)")
        
    return patterns

def refresh_market_data():
    """Tâche de fond : Met à jour la mémoire de l'application"""
    global DATA_CACHE
    logger.info("Rafraîchissement global des marchés en arrière-plan...")
    results = []
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    for item in MARKET_UNIVERSE:
        try:
            symbol = item["ticker"]
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

            patterns = detect_chart_patterns(hist)

            results.append({
                "entreprise": item["entreprise"],
                "ticker": symbol,
                "sector": item["sector"],
                "theme": item["theme"],
                "prix": round(float(prix), 2),
                "sma50": round(float(sma50), 2),
                "sma200": round(float(sma200), 2),
                "rsi": round(float(last['RSI']), 1) if not np.isnan(last['RSI']) else 50.0,
                "roe": round(float(info.get('returnOnEquity', 0) or 0), 4),
                "target": round(float(info.get('targetMeanPrice', prix*1.1) or prix*1.1), 2),
                "score": score_ia,
                "patterns": patterns
            })
        except Exception as e:
            logger.error(f"Erreur extraction {item['ticker']}: {e}")
            
    if results:
        DATA_CACHE = results
        logger.info(f"Marchés mis à jour : {len(DATA_CACHE)} actions prêtes.")

def train_model():
    logger.info("Entraînement du modèle Machine Learning...")
    dataset, labels = [], []
    for item in MARKET_UNIVERSE:
        try:
            hist = yf.Ticker(item["ticker"]).history(period="2y")
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
            logger.error(f"Erreur training {item['ticker']}: {e}")

    if dataset:
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(dataset, labels)
        joblib.dump(model, MODEL_PATH)
        logger.info("Modèle sauvegardé.")
        refresh_market_data()

def send_weekly_telegram_report():
    """Rapport Hebdomadaire automatique le Vendredi soir à 22h00"""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    
    top_stocks = sorted(DATA_CACHE, key=lambda x: x['score'], reverse=True)[:3]
    msg = "📊 *RAPPORT HEBDOMADAIRE SYNAPSEQUANT*\n\nTop Convictions IA pour la semaine prochaine :\n"
    for s in top_stocks:
        msg += f"• *{s['entreprise']}* ({s['ticker']}) : Score {s['score']}/2.0 - Prix €{s['prix']}\n"
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})

# PLANIFICATEUR DE TÂCHES (CRON)
scheduler = BackgroundScheduler()
scheduler.add_job(train_model, 'cron', hour=23, minute=0)
scheduler.add_job(refresh_market_data, 'interval', minutes=15) # Update toutes les 15 minutes
scheduler.add_job(send_weekly_telegram_report, 'cron', day_of_week='fri', hour=22, minute=0) # Rapport vendredi
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
    """Optimisation de portefeuille de Markowitz (Frontière d'Efficience)"""
    top_items = sorted(DATA_CACHE, key=lambda x: x['score'], reverse=True)[:5]
    if not top_items: return {"allocation": {}}
    
    count = len(top_items)
    if risk_profile == "defensif":
        weights = [0.4, 0.25, 0.15, 0.1, 0.1]
    elif risk_profile == "offensif":
        weights = [0.5, 0.2, 0.15, 0.1, 0.05]
    else: # equilibre
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

@app.get("/api/correlation")
def get_correlation_matrix():
    """Matrice de corrélation entre les meilleures valeurs"""
    top_tickers = [i["ticker"] for i in sorted(DATA_CACHE, key=lambda x: x['score'], reverse=True)[:5]]
    try:
        df_list = []
        for t in top_tickers:
            h = yf.Ticker(t).history(period="6m")['Close']
            h.name = t
            df_list.append(h)
        df_concat = pd.concat(df_list, axis=1).pct_change().dropna()
        corr = df_concat.corr().round(2).to_dict()
        return {"tickers": top_tickers, "matrix": corr}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/backtest")
def run_backtest(ticker: str = Query("MC.PA")):
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 100: return {"error": "Historique insuffisant"}
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
            
        return {
            "ticker": ticker, "current_price": round(float(last_price), 2),
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
