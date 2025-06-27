import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
import datetime
import ta
import hashlib

# -------------- ADMIN LOGIN CONFIG -------------------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

HASHED_PASSWORD = hash_password(ADMIN_PASSWORD)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Admin Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login = st.form_submit_button("Login")

        if login:
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.success("✅ Login successful!")
                st.session_state.authenticated = True
                st.write("🔁 Please refresh the page to proceed.")
                st.stop()
            else:
                st.error("❌ Incorrect username or password.")
    st.stop()

# -------------- MAIN APP -------------------
st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Price Movement Predictor")

tab1, tab2, tab3 = st.tabs(["Predict", "Backtest", "Chatbot"])

def load_and_process_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if data.empty:
        st.warning(f"No data returned for {ticker}. Check the ticker.")
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if 'Close' not in data.columns:
        st.error("No valid 'Close' price data found.")
        return pd.DataFrame()

    # Technical indicators
    data['MA5'] = data['Close'].rolling(5).mean()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['Volume_Change'] = data['Volume'].pct_change()

    close_series = data['Close'].squeeze()
    data['RSI'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
    macd = ta.trend.MACD(close=close_series)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=close_series)
    data['BB_Width'] = bb.bollinger_wband()

    return data.dropna()

# ----------------- TAB 1: Predict -----------------
with tab1:
    st.subheader("📊 Predict Stock Movement")
    ticker = st.text_input("Enter Stock Ticker:", "AAPL")
    if st.button("Predict"):
        end = datetime.datetime.today()
        start = end - datetime.timedelta(days=365 * 2)
        data = load_and_process_data(ticker, start, end)

        if not data.empty:
            st.line_chart(data[['Close', 'MA5', 'MA10']])
            latest = data.iloc[-1:]

            features = ['Close', 'MA5', 'MA10', 'Volume_Change', 'RSI', 'MACD', 'MACD_Signal', 'BB_Width']
            model = joblib.load("stock_model.pkl")
            prediction = model.predict(latest[features])[0]
            prob = model.predict_proba(latest[features])[0][prediction]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"⬆️ Stock likely to go UP ({prob:.2%} confidence)")
            else:
                st.warning(f"⬇️ Stock likely to go DOWN ({prob:.2%} confidence)")

# ----------------- TAB 2: Backtest -----------------
with tab2:
    st.subheader("🔄 Backtesting Simulator")
    ticker = st.text_input("Backtest Ticker:", "AAPL", key="bt")
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365 * 2)
    data = load_and_process_data(ticker, start, end)

    if not data.empty:
        features = ['Close', 'MA5', 'MA10', 'Volume_Change', 'RSI', 'MACD', 'MACD_Signal', 'BB_Width']
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        model = joblib.load("stock_model.pkl")
        data['Prediction'] = model.predict(data[features])
        accuracy = (data['Prediction'] == data['Target']).mean()

        st.metric("Backtest Accuracy", f"{accuracy:.2%}")
        st.line_chart(data[['Close']])

# ----------------- TAB 3: Chatbot -----------------
with tab3:
    st.subheader("💬 Stock Assistant Chatbot")
    user_input = st.text_input("Ask a question about technical indicators or trading:")

    if user_input:
        response = ""

        # Lowercase and keyword matching
        question = user_input.lower()

        # --- Beginner ---
        if "what is" in question and "stock" in question:
            response = "A stock represents ownership in a company and a claim on its earnings and assets."
        elif "moving average" in question:
            response = "Moving Averages smooth price data to help identify the direction of a trend. MA5 is 5-day average, MA10 is 10-day."
        elif "rsi" in question:
            response = "RSI (Relative Strength Index) measures momentum. RSI > 70 = overbought, < 30 = oversold."
        elif "macd" in question:
            response = "MACD (Moving Average Convergence Divergence) shows momentum and trend direction using EMA differences."

        # --- Intermediate ---
        elif "bollinger" in question:
            response = "Bollinger Bands show volatility. When bands widen, volatility increases. Price near upper band may signal overbought."
        elif "volume change" in question:
            response = "Volume Change tracks buying/selling pressure. Sudden spikes can indicate big trader actions."

        # --- Advanced ---
        elif "strategy" in question or "backtest" in question:
            response = "Backtesting is simulating your trading strategy on past data to evaluate performance before using real money."
        elif "overfitting" in question:
            response = "Overfitting is when a model performs well on training data but poorly on new, unseen data."

        else:
            response = "🤖 I'm a simple chatbot. Ask about indicators like RSI, MACD, MA, Bollinger Bands, or backtesting concepts."

        st.write(response)
