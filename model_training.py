import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import ta

# 📈 List of tickers to train on
tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'INFY.NS', 'RELIANCE.NS']

all_data = []

for ticker in tickers:
    print(f"📥 Downloading data for {ticker}")
    data = yf.download(ticker, start='2020-01-01', end='2024-01-01', auto_adjust=False)

    # Flatten MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Skip if empty
    if data.empty:
        print(f"⚠️ No data for {ticker}, skipping.")
        continue

    # 📊 Feature engineering
    data['Ticker'] = ticker
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['Volume_Change'] = data['Volume'].pct_change()

    close_series = data['Close'].squeeze()
    data['RSI'] = ta.momentum.RSIIndicator(close=close_series).rsi()
    macd = ta.trend.MACD(close=close_series)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=close_series)
    data['BB_Width'] = bb.bollinger_wband()

    # 🎯 Target
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

    # 🧼 Drop missing values
    data = data.dropna()

    # Add to master list
    all_data.append(data)

# 🔗 Combine all data
if not all_data:
    print("❌ No data to train on.")
    exit()

df = pd.concat(all_data)

# ✅ Define features
features = ['Close', 'MA5', 'MA10', 'Volume_Change', 'RSI', 'MACD', 'MACD_Signal', 'BB_Width']
X = df[features].copy()
X.columns = [col.strip() for col in X.columns]  # Remove spaces
X = X[features]  # Ensure correct order
y = df['Target']

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 🧠 Train model
print("🧠 Training model...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 💾 Save model
joblib.dump(model, "stock_model.pkl")
print("✅ Model saved as stock_model.pkl")

# 📊 Accuracy
accuracy = model.score(X_test, y_test)
print(f"📈 Accuracy: {accuracy:.2%}")

# 📈 Feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(features, model.feature_importances_)
plt.title("📊 Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.grid(True)
plt.show()