from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf

app = FastAPI(title="Terminal Quantitatif ML")

# Autorise les requêtes externes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connecte le dossier contenant votre HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# Votre univers d'investissement (CAC40, US Tech, Europe, Asie)
TICKERS = {
    "LVMH": "MC.PA", "TotalEnergies": "TTE.PA", "Air Liquide": "AI.PA", "Airbus": "AIR.PA",
    "Sanofi": "SAN.PA", "L'Oréal": "OR.PA", "Hermès": "RMS.PA", "Schneider Electric": "SU.PA",
    "Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA", "Tesla": "TSLA",
    "Amazon": "AMZN", "Meta": "META", "Alphabet": "GOOGL", "Broadcom": "AVGO",
    "ASML": "ASML.AS", "SAP": "SAP.DE", "Siemens": "SIE.DE", "Toyota": "7203.T"
}

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/data")
def get_ml_data():
    results = []
    for name, symbol in TICKERS.items():
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            prix = info.get('currentPrice', info.get('regularMarketPrice', 0))
            sma50 = info.get('fiftyDayAverage', 0)
            sma200 = info.get('twoHundredDayAverage', 0)
            marge = info.get('profitMargins', 0)
            target = info.get('targetMeanPrice', 0)
            perf_52w = info.get('52WeekChange', 0)
            
            # Modèle de scoring déterministe (avant intégration Scikit-Learn)
            score = 0
            if prix and sma200 and prix > sma200: score += 0.5
            if prix and sma50 and prix < sma50: score += 0.5
            if target and prix and target > (prix * 1.1): score += 1.0
            
            results.append({
                "entreprise": name,
                "ticker": symbol,
                "prix": prix,
                "perf": perf_52w,
                "sma50": sma50,
                "sma200": sma200,
                "marge": marge,
                "target": target,
                "score": score
            })
        except Exception as e:
            print(f"Erreur sur {symbol}: {e}")
            
    return results
