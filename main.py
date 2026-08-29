from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import threading
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QuantEngine")

app = FastAPI(title="SynapseQuant Engine - Terminal Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Univers d'investissements
TICKERS = {
    "LVMH": "MC.PA", "TotalEnergies": "TTE.PA", "BNP Paribas": "BNP.PA", "Apple": "AAPL",
    "Microsoft": "MSFT", "ASML": "ASML.AS", "JPMorgan": "JPM", "ExxonMobil": "XOM"
}

# Tickers Macro & Matières Premières
MACRO_TICKERS = {
    "Oil": "CL=F",      # Pétrole WTI
    "Gold": "GC=F",     # Or
    "VIX": "^VIX",      # Indice de peur / Volatilité
    "US10Y": "^TNX"     # Taux 10 ans US
}

MODEL_PATH = "ml_model.pkl"

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs techniques clés (RSI, MACD, Volatilité, Ratios)."""
    df = df.copy()
    
    # Moyennes Mobiles
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # RSI (14 jours)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    
    # Volatilité historique (std des rendements sur 20j)
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(20).std()
    
    # Ratio de Volume (Volume du jour / Moyenne 50j)
    df['Vol_Ratio'] = df['Volume'] / (df['Volume'].rolling(50).mean() + 1e-9)
    
    return df

def train_model():
    """Entraînement Machine Learning multi-factoriel (Technique + Macro)."""
    logger.info("Démarrage de l'entraînement quantitatif du modèle ML...")
    dataset = []
    labels = []
    
    try:
        # 1. Extraction des séries Macro/Matières premières
        macro_data = {}
        for key, sym in MACRO_TICKERS.items():
            h = yf.Ticker(sym).history(period="2y")
            if not h.empty:
                macro_data[key] = h['Close']
        macro_df = pd.DataFrame(macro_data).ffill().bfill()
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des données Macro: {e}")
        macro_df = pd.DataFrame()

    # 2. Construction du Dataset d'entraînement pour chaque action
    for name, symbol in TICKERS.items():
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="2y")
            if len(hist) < 200:
                continue
            
            hist = compute_technical_indicators(hist)
            
            # Alignement temporel des données Macro avec l'action
            if not macro_df.empty:
                hist = hist.join(macro_df, how='left').ffill().bfill()
            else:
                hist['Oil'] = 75.0
                hist['VIX'] = 18.0

            hist = hist.dropna()
            
            # Cible (Target) : Hausse de +2% à un horizon de 15 jours de bourse
            hist['Target'] = np.where(hist['Close'].shift(-15) > hist['Close'] * 1.02, 1, 0)
            
            for index, row in hist.iterrows():
                # Matrice de Features (8 variables d'entrée)
                features = [
                    row['Close'] / (row['SMA50'] + 1e-9),
                    row['Close'] / (row['SMA200'] + 1e-9),
                    row['RSI'] / 100.0,
                    row['Volatility'] if not np.isnan(row['Volatility']) else 0.01,
                    row['Vol_Ratio'] if not np.isnan(row['Vol_Ratio']) else 1.0,
                    row['MACD'],
                    row['Oil'] / 100.0,
                    row['VIX'] / 50.0
                ]
                dataset.append(features)
                labels.append(row['Target'])
        except Exception as e:
            logger.error(f"Erreur traitement entraînement pour {symbol}: {e}")
            continue

    if dataset:
        # Modèle Random Forest robuste (300 arbres, profondeur limitée pour éviter le surapprentissage)
        model = RandomForestClassifier(
            n_estimators=300, 
            max_depth=10, 
            min_samples_leaf=5, 
            random_state=42
        )
        model.fit(dataset, labels)
        joblib.dump(model, MODEL_PATH)
        logger.info("Modèle ML ré-entraîné et sauvegardé avec succès.")

# Planificateur de tâches : Re-entraînement quotidien à 23h00
scheduler = BackgroundScheduler()
scheduler.add_job(train_model, 'cron', hour=23, minute=0)
scheduler.start()

@app.on_event("startup")
def startup_event():
    """Initialisation au démarrage du serveur."""
    if not os.path.exists(MODEL_PATH):
        threading.Thread(target=train_model).start()

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/data")
def get_ml_data():
    results = []
    
    # Chargement du modèle Random Forest
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    # Extraction des données Macro en direct
    latest_oil = 75.0
    latest_vix = 18.0
    try:
        oil_hist = yf.Ticker(MACRO_TICKERS["Oil"]).history(period="5d")
        vix_hist = yf.Ticker(MACRO_TICKERS["VIX"]).history(period="5d")
        if not oil_hist.empty: latest_oil = oil_hist['Close'].iloc[-1]
        if not vix_hist.empty: latest_vix = vix_hist['Close'].iloc[-1]
    except Exception as e:
        logger.warning(f"Impossible de récupérer les données Macro Live: {e}")

    for name, symbol in TICKERS.items():
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1y")
            
            if hist.empty:
                continue

            hist = compute_technical_indicators(hist)
            last_row = hist.iloc[-1]
            
            prix = info.get('currentPrice', info.get('regularMarketPrice', last_row['Close']))
            sma50 = last_row['SMA50'] if not np.isnan(last_row['SMA50']) else prix
            sma200 = last_row['SMA200'] if not np.isnan(last_row['SMA200']) else prix
            
            peg = info.get('pegRatio', 0) or 0
            pb = info.get('priceToBook', 0) or 0
            roe = info.get('returnOnEquity', 0) or 0
            target = info.get('targetMeanPrice', 0) or (prix * 1.1)
            
            score_ia = 1.0 # Score neutre par défaut
            
            if model and not np.isnan(sma50) and not np.isnan(sma200):
                features_live = [[
                    prix / (sma50 + 1e-9),
                    prix / (sma200 + 1e-9),
                    (last_row['RSI'] if not np.isnan(last_row['RSI']) else 50.0) / 100.0,
                    last_row['Volatility'] if not np.isnan(last_row['Volatility']) else 0.01,
                    last_row['Vol_Ratio'] if not np.isnan(last_row['Vol_Ratio']) else 1.0,
                    last_row['MACD'] if not np.isnan(last_row['MACD']) else 0.0,
                    latest_oil / 100.0,
                    latest_vix / 50.0
                ]]
                # Probabilité prédite par l'IA de faire +2% à 15 jours
                proba = model.predict_proba(features_live)[0][1]
                score_ia = round(proba * 2.0, 2)
                
                # Pondération fondamentale (post-filtrage prudentif)
                if roe > 0.15: score_ia = min(2.0, score_ia + 0.1)
                if peg > 0 and peg < 1.0: score_ia = min(2.0, score_ia + 0.1)
                if pb > 5.0: score_ia = max(0.0, score_ia - 0.15)
            else:
                # Calcul algorithmique de secours
                if prix > sma200: score_ia += 0.3
                if roe > 0.15: score_ia += 0.3
                if peg > 0 and peg < 1.2: score_ia += 0.4

            results.append({
                "entreprise": name,
                "ticker": symbol,
                "prix": round(float(prix), 2),
                "sma50": round(float(sma50), 2),
                "sma200": round(float(sma200), 2),
                "peg": round(float(peg), 2),
                "pb": round(float(pb), 2),
                "roe": round(float(roe), 4),
                "target": round(float(target), 2),
                "score": round(float(score_ia), 2)
            })
        except Exception as e:
            logger.error(f"Erreur traitement Live pour {symbol}: {e}")
            
    return results
