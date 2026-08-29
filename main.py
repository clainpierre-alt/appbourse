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

app = FastAPI(title="Terminal Quantitatif ML")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

TICKERS = {
    "LVMH": "MC.PA", "TotalEnergies": "TTE.PA", "BNP Paribas": "BNP.PA", "Apple": "AAPL",
    "Microsoft": "MSFT", "ASML": "ASML.AS", "JPMorgan": "JPM", "ExxonMobil": "XOM"
}

MODEL_PATH = "ml_model.pkl"

def train_model():
    """Fonction d'apprentissage continu : s'exécute en arrière-plan chaque nuit"""
    print("Démarrage de l'entraînement ML...")
    dataset = []
    labels = []
    
    for name, symbol in TICKERS.items():
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="2y")
            if len(hist) < 200: continue
            
            # Création de données d'entraînement synthétiques basées sur l'historique
            # Dans un cas réel, on décale les prix pour prédire J+30
            hist['SMA50'] = hist['Close'].rolling(50).mean()
            hist['SMA200'] = hist['Close'].rolling(200).mean()
            hist = hist.dropna()
            
            for index, row in hist.iterrows():
                # Features techniques historiques simplifiées
                features = [
                    row['Close'] / row['SMA50'], 
                    row['Close'] / row['SMA200'],
                    row['Volume']
                ]
                # Label : 1 si le prix a monté le jour suivant, 0 sinon
                # (Simplification pour l'exemple d'architecture)
                dataset.append(features)
                labels.append(1 if row['Close'] > row['Open'] else 0)
        except:
            continue

    if dataset:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(dataset, labels)
        joblib.dump(model, MODEL_PATH)
        print("Modèle ré-entraîné et sauvegardé avec succès.")

# Planificateur de tâches : Entraînement tous les jours à 23h00
scheduler = BackgroundScheduler()
scheduler.add_job(train_model, 'cron', hour=23, minute=0)
scheduler.start()

@app.on_event("startup")
def startup_event():
    """Lance un premier entraînement asynchrone si aucun modèle n'existe"""
    if not os.path.exists(MODEL_PATH):
        threading.Thread(target=train_model).start()

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/data")
def get_ml_data():
    results = []
    
    # Chargement du modèle s'il existe
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    for name, symbol in TICKERS.items():
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Extraction des fondamentaux inter-sectoriels
            prix = info.get('currentPrice', info.get('regularMarketPrice', 0))
            sma50 = info.get('fiftyDayAverage', 0)
            sma200 = info.get('twoHundredDayAverage', 0)
            peg = info.get('pegRatio', 0) # Croissance (Tech)
            pb = info.get('priceToBook', 0) # Valorisation (Banque)
            dte = info.get('debtToEquity', 0) # Risque (Industrie)
            roe = info.get('returnOnEquity', 0)
            target = info.get('targetMeanPrice', 0)
            
            score_ia = 0
            if model and sma50 and sma200:
                # Prédiction via le modèle ML entraîné
                features_live = [[prix / sma50, prix / sma200, info.get('averageVolume', 0)]]
                proba = model.predict_proba(features_live)[0][1] # Probabilité de hausse
                score_ia = round(proba * 2.0, 2) # Mise à l'échelle sur 2.0
            else:
                # Score de secours algorithmique
                if prix > sma200: score_ia += 0.5
                if roe and roe > 0.15: score_ia += 0.5
                if pb and pb < 2: score_ia += 0.5
            
            results.append({
                "entreprise": name, "ticker": symbol, "prix": prix,
                "sma50": sma50, "sma200": sma200, "peg": peg, "pb": pb, 
                "roe": roe, "target": target, "score": score_ia
            })
        except Exception as e:
            print(f"Erreur extraction {symbol}")
            
    return results
