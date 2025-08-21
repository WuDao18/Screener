#current verion updated 18/4/25
#include weekly and all countries

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import tempfile
import firebase_admin
from firebase_admin import credentials, auth,firestore
from datetime import datetime, timedelta
import random
import requests

if not firebase_admin._apps:
    firebase_creds = dict(st.secrets["firebase_credentials"])
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

db = firestore.client()
print("Firebase initialized successfully!")

def get_latest_date():
    """Fetch the last updated date from Firestore (only once)."""
    selected_exchange = st.session_state.get("selected_exchange", "MYX")
    timeframe = st.session_state.get("period","Daily")
    if timeframe == "Weekly":
        doc_ref = db.collection("data_date").document(f"latest_date_{selected_exchange}_Weekly")
        doc = doc_ref.get()
    else:
        doc_ref = db.collection("data_date").document(f"latest_date_{selected_exchange}")
        doc = doc_ref.get()

    if doc.exists:
      #  print(f"Last recorded date: {doc.to_dict().get('date')}")
        return doc.to_dict().get('date')
    else:
        print("No date found.")

def send_otp(email):
    """Generate, store, and send OTP to user's Telegram."""
    TELEGRAM_BOT_TOKEN = st.secrets["telegram"]["BOT_TOKEN"]
    user_ref = db.collection("users").document(email)
    user = user_ref.get()

    if user.exists:
        user_data = user.to_dict()
        telegram_chat_id = user_data.get("chat_id")  # Retrieve Telegram chat ID

        if not telegram_chat_id:
            st.error("No Telegram chat ID found for this user. Please register your Telegram ID.")
            return

        otp = str(random.randint(100000, 999999))  # Generate a 6-digit OTP
        expiration_time = datetime.utcnow() + timedelta(minutes=5)  # OTP valid for 5 minutes

        # ✅ Store OTP with email as document ID
        otp_ref = db.collection("otp_verifications").document(email)
        otp_ref.set({
            "otp": otp,
            "expiration_time": expiration_time.isoformat(),
            "verified": False
        })

        # ✅ Send OTP to user's Telegram
        message = f"🔐 Your OTP code is: {otp}\n⚠️ This OTP expires in 5 minutes."
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": telegram_chat_id,
            "text": message
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            st.success("OTP 已经发出到您的 Telegram。OTP has been sent to your Telegram.")
        else:
            st.error("Failed to send OTP via Telegram. Please try again.")
    else:
        st.error("Email not found in Firestore. Please register first.")

def validate_otp(email, otp):
    """Validate the OTP provided by the user."""
    try:
        # Normalize email to lowercase for case-insensitive matching
        normalized_email = email.strip().lower()

        otp_doc = db.collection("otp_verifications").document(normalized_email).get()
        if not otp_doc.exists:
            print("No OTP request found for this user.")
            return False

        otp_data = otp_doc.to_dict()
        stored_otp = str(otp_data["otp"])  # Ensure stored OTP is a string
        expiration_time_str = otp_data["expiration_time"]

        print(f"Database OTP is: {stored_otp}")
        print(f"User entered OTP is: {otp}")

        # Convert expiration time properly
        try:
            expiration_time = datetime.fromisoformat(expiration_time_str)  # If stored as ISO string
        except ValueError:
            expiration_time = otp_data["expiration_time"]  # If stored as Firestore timestamp

        # Check if OTP matches and has not expired
        if datetime.utcnow() > expiration_time:
            print("OTP has expired. 密码逾期")
            return False

        if str(otp) == stored_otp:  # Ensure both are compared as strings
            # Mark the OTP as verified
            db.collection("otp_verifications").document(email).update({"verified": True})
            print("OTP validated successfully! 成功验证！")
            return True
        else:
            print("Invalid OTP. 密码错误。")
            return False

    except Exception as e:
        print(f"Error validating OTP: {e}")
        return False

def add_custom_css():
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: #007BFF;
            color: white;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: blue;
            color: white;
        }
        </style>

        <style>
        /* Style for the download button */
        div.stDownloadButton > button {
            background-color: #004080; 
            color: white;
            font-weight: bold;

        }
        div.stDownloadButton > button:hover {
            background-color: #002b5e; 
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)


st.markdown(
    """
    <style>
    /* Expanded sidebar */
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 800px !important; /* Force expanded sidebar width */
        min-width: 800px !important; /* Ensure minimum width */
        transition: width 0.3s ease-in-out;
    }
    [data-testid="stSidebar"][aria-expanded="true"] .sidebar-content {
        width: 800px !important; /* Match the expanded sidebar width */
    }

    /* Minimized sidebar */
    [data-testid="stSidebar"][aria-expanded="false"] {
        width: 5px !important; /* Force minimized sidebar width */
        min-width: 5px !important; /* Ensure minimum width */
        transition: width 0.3s ease-in-out;
    }
    [data-testid="stSidebar"][aria-expanded="false"] .sidebar-content {
        width: 5px !important; /* Match the minimized sidebar width */
    }

    /* Main content adjustment */
    [data-testid="stSidebar"][aria-expanded="true"] ~ .main {
        margin-left: 800px !important; /* Adjust main content for expanded sidebar */
        transition: margin-left 0.3s ease-in-out;
    }
    [data-testid="stSidebar"][aria-expanded="false"] ~ .main {
        margin-left: 5px !important; /* Adjust main content for minimized sidebar */
        transition: margin-left 0.3s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#change dropdown menu font size and colour
st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div {
        font-size: 16px !important;
        font-family: Arial, sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# color: darkblue !important;
def download_file():
    """Retrieve all stock data from the `screened_result` collection and return as a DataFrame."""
    selected_exchange = st.session_state.get("selected_exchange", "MYX")  # Default to MYX if not set
    timeframe = st.session_state.get("period", "Daily")

    if timeframe == "Weekly":
        collection_ref = db.collection(f"screened_result_{selected_exchange}_Weekly")
        docs = collection_ref.stream()

    else:
        collection_ref = db.collection(f"screened_result_{selected_exchange}")
        docs = collection_ref.stream()

    stock_list = []
    for doc in docs:
        stock_data = doc.to_dict()
        stock_list.append(stock_data)

    if not stock_list:
        print(f"No data found in Firestore collection: screened_result_{selected_exchange}_{timeframe}.")
        return None

    # Convert list of stock data to Pandas DataFrame
    df = pd.DataFrame(stock_list)
    return df

def load_stock_data(stock_symbol):
    """Load stock data from Firestore and return a Pandas DataFrame."""
    selected_exchange = st.session_state.get("selected_exchange", "MYX")  # Default to MYX if not set
    timeframe = st.session_state.get("period", "Daily")

    if timeframe == "Weekly":
        doc_ref = db.collection(f"stocks_{selected_exchange}_Weekly").document(stock_symbol)  # Reference to Firestore document
        doc = doc_ref.get()  # Retrieve document

    else:
        doc_ref = db.collection(f"stocks_{selected_exchange}").document(stock_symbol)  # Reference to Firestore document
        doc = doc_ref.get()  # Retrieve document

    if doc.exists:
        stock_data = doc.to_dict().get("data", [])  # Get the 'data' list
        if stock_data:
            df = pd.DataFrame(stock_data)  # Convert list of dictionaries to DataFrame
            df["date"] = pd.to_datetime(df["date"])  # Ensure date is in datetime format
            df = df.set_index("date")  # Set 'date' as the index
            return df
        else:
            st.error(f"No data available for {stock_symbol}- {timeframe}.")
            return None
    else:
        st.error(f"Stock {stock_symbol} - {timeframe} not found in Firestore.")
        return None

def display_symbols_dropdown(symbols):
    selected_stock = st.selectbox("请选择一只股票查看图表： Select a stock to view the chart:", options=symbols)
    if st.button("看图表 Show Chart"):
        select_stock(selected_stock)
        st.session_state['selected_stock'] = selected_stock

# Function to handle stock selection
def select_stock(stock):
    st.session_state['selected_stock'] = stock


def check_indicators_and_save(df, min_volume, min_price, brsi_value, brsi2_value, hrsi_value, rrsi_value, rsi_min, rsi_max):
    try:
        # Initialize mask for filtering
        mask = pd.Series([True] * len(df))

        # Apply filters based on checkbox states
        if st.session_state.get('r1_check', False):
            mask &= (df['r1'] == 1)
        if st.session_state.get('r2_check', False):
            mask &= (df['r2'] == 1)
        if st.session_state.get('r3_check', False):
            mask &= (df['r3'] == 1)
        if st.session_state.get('n1_check', False):
            mask &= (df['n1'] == 1)
        if st.session_state.get('x9_check', False):
            mask &= (df['x9'] == 1)
        if st.session_state.get('y1_check', False):
            mask &= (df['y1'] == 1)
        if st.session_state.get('dxw_check', False):
            mask &= (df['dxw'] == 1)
        if st.session_state.get('vm_check', False):
            mask &= (df['volumeM'] == 1)
        if st.session_state.get('vm2_check', False):
            mask &= (df['volumeM2'] == 1)
        if st.session_state.get('pmc_check', False):
            mask &= (df['priceMC'] == 1)
        if st.session_state.get('pmh_check', False):
            mask &= (df['priceMH'] == 1)
        if st.session_state.get('gc_check', False):
            mask &= (df['MACDGC'] == 1)
        if st.session_state.get('dc_check', False):
            mask &= (df['MACDDC'] == 1)
        if st.session_state.get('a0_check', False):
            mask &= (df['MACDA0'] == 1)
        if st.session_state.get('b0_check', False):
            mask &= (df['MACDB0'] == 1)
        if st.session_state.get('DKWR_check', False):
            mask &= (df['DKW'] == 1)
        if st.session_state.get('DKWB_check', False):
            mask &= (df['DKW'] == 2)
        if st.session_state.get('R2B_check', False):
            mask &= (df['DKW_R2B'] == 1)
        if st.session_state.get('B2R_check', False):
            mask &= (df['DKW_B2R'] == 1)
        if st.session_state.get('ZLB0_check', False):
            mask &= (df['zlB0'] == 1)
        if st.session_state.get('ZLR_check', False):
            mask &= (df['zlR'] == 1)
        if st.session_state.get('ZLG2R_check', False):
            mask &= (df['zlG2R'] == 1)
        if st.session_state.get('ZLR2G_check', False):
            mask &= (df['zlR2G'] == 1)
        if st.session_state.get('brsiMma_check', False):
            mask &= (df['brsiMma'] == 1)
        if st.session_state.get('brsi1Mma_check', False):
            mask &= (df['brsi1stMma'] == 1)
        if st.session_state.get('rrsi1_check', False):
            mask &= (df['rrsi1z'] == 1)
        if st.session_state.get('zj_check', False):
            mask &= (df['zj'] == 1)
        if st.session_state.get('qs_rbpl_check', False):
            mask &= (df['qs'] == 1)
        if st.session_state.get('qs_grb_check', False):
            mask &= (df['qsGRB'] == 1)
        if st.session_state.get('qs_rgb_check', False):
            mask &= (df['qsRGB'] == 1)
        if st.session_state.get('qs_gpl_check', False):
            mask &= (df['qsGPL'] == 1)
        if st.session_state.get('qs_pgl_check', False):
            mask &= (df['qsPGL'] == 1)
        if st.session_state.get('qs_atm_check', False):
            mask &= (df['qsATM'] == 1)
        if st.session_state.get('qs_atmS_check', False):
            mask &= (df['qsATMS'] == 1)
        if st.session_state.get('qs_rbd_check', False):
            qs_selection = st.session_state.get('qs_rbd_check', 0)
            if qs_selection == 2:
                mask &= ((df['qs2R'] == 1)| (df['qsGRB'] == 1))
            elif qs_selection == 3:
                mask &= ((df['qs3R'] == 1)| (df['qs2R'] == 1)| (df['qsGRB'] == 1))
            elif qs_selection == 4:
                mask &= ((df['qs4R'] == 1)|(df['qs3R'] == 1) | (df['qs2R'] == 1)| (df['qsGRB'] == 1))
            elif qs_selection == 5:
                mask &= ((df['qs5R'] == 1) | (df['qs4R'] == 1)|(df['qs3R'] == 1) | (df['qs2R'] == 1)| (df['qsGRB'] == 1))
            # if qs_selection == 2:
            #     mask &= (df['qs2R'] == 1)
            # elif qs_selection == 3:
            #     mask &= (df['qs3R'] == 1)
            # elif qs_selection == 4:
            #     mask &= (df['qs4R'] == 1)
            # elif qs_selection == 5:
            #     mask &= (df['qs5R'] == 1)
        if st.session_state.get('zj_R2G_check', False):
            mask &= (df['zjR2G'] == 1)
        if st.session_state.get('zj_G2R_check', False):
            mask &= (df['zjG2R'] == 1)
        if st.session_state.get('zj_rbd_check', False):
            qs_selection = st.session_state.get('zj_rbd_check', 0)
            if qs_selection == 2:
                mask &= ((df['zj2R'] == 1)| (df['zjG2R'] == 1))
            elif qs_selection == 3:
                mask &= ((df['zj3R'] == 1)| (df['zj2R'] == 1)| (df['zjG2R'] == 1))
            elif qs_selection == 4:
                mask &= ((df['zj4R'] == 1)|(df['zj3R'] == 1) | (df['zj2R'] == 1)| (df['zjG2R'] == 1))
            elif qs_selection == 5:
                mask &= ((df['zj5R'] == 1) | (df['zj4R'] == 1)|(df['zj3R'] == 1) | (df['zj2R'] == 1)| (df['zjG2R'] == 1))
        if st.session_state.get('DKWRD_check', False):
            dkwr_selection = st.session_state.get('DKWRD_check', 0)
            if dkwr_selection == 2:
                mask &= (df['DKW2R'] == 1)
            elif dkwr_selection == 3:
                mask &= (df['DKW3R'] == 1)
            elif dkwr_selection == 4:
                mask &= (df['DKW4R'] == 1)
            elif dkwr_selection == 5:
                mask &= (df['DKW5R'] == 1)
        if st.session_state.get('DKWBD_check', False):
            dkwb_selection = st.session_state.get('DKWBD_check', 0)
            if dkwb_selection == 2:
                mask &= (df['DKW2B'] == 1)
            elif dkwb_selection == 3:
                mask &= (df['DKW3B'] == 1)
            elif dkwb_selection == 4:
                mask &= (df['DKW4B'] == 1)
            elif dkwb_selection == 5:
                mask &= (df['DKW5B'] == 1)
        if st.session_state.get('brsi_operator', False):
            operator_selection = st.session_state.get('brsi_operator', 0)
            if operator_selection == ">=":
                mask &= (df['brsi'] >= brsi_value )
            elif operator_selection == "<=":
                mask &= (df['brsi'] <= brsi_value)
            elif operator_selection == "=":
                mask &= (df['brsi'] == brsi_value)
        if st.session_state.get('brsi2_operator', False):
            operator_selection = st.session_state.get('brsi2_operator', 0)
            if operator_selection == ">=":
                mask &= (df['brsi'] >= brsi2_value )
            elif operator_selection == "<=":
                mask &= (df['brsi'] <= brsi2_value)
            elif operator_selection == "=":
                mask &= (df['brsi'] == brsi2_value)
        if st.session_state.get('hrsi_operator', False):
            operator_selection = st.session_state.get('hrsi_operator', 0)
            if operator_selection == ">=":
                mask &= (df['hrsi'] >= hrsi_value )
            elif operator_selection == "<=":
                mask &= (df['hrsi'] <= hrsi_value)
            elif operator_selection == "=":
                mask &= (df['hrsi'] == hrsi_value)
        if st.session_state.get('rrsi_operator', False):
            operator_selection = st.session_state.get('rrsi_operator', 0)
            if operator_selection == ">=":
                mask &= (df['rrsi'] >= rrsi_value )
            elif operator_selection == "<=":
                mask &= (df['rrsi'] <= rrsi_value)
            elif operator_selection == "=":
                mask &= (df['rrsi'] == rrsi_value)

        # Apply user-defined filters
        if min_volume:
            mask &= (df['volume'] >= min_volume * 100000)
        if min_price:
            mask &= (df['close'] >= min_price)
        if rsi_min and st.session_state.get('rsi_check', False):
            mask &= (df['rsi'] >= rsi_min)
        if rsi_max and st.session_state.get('rsi_check', False):
            mask &= (df['rsi'] <= rsi_max)

        # Get matching symbols
        matching_symbols = [str(symbol) for symbol in df[mask]['symbol'].tolist()]

        # Create a temporary file to save the filtered results
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as temp_file:
            # Write the matching symbols to the file
            temp_file.write("\n".join(matching_symbols))
            temp_file_path = temp_file.name  # Get the path for the saved file

        # Return the matching symbols and file path for further use (could be used to read back or process further)
        return matching_symbols, temp_file_path

    except Exception as e:
        # Add more detailed error info
        st.error(f"Error processing data: {str(e)}, Data type of the DataFrame: {type(df)}")
        return [], None

def weight(values, length):
    wma = values.rolling(window=length).apply(lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)))
    sma = values.rolling(window=length).mean()
    return wma * 3 - sma * 2

def calculate_trend_data(df):
    # Calculate price trend
    price_trend = (df['close'] * 3 + df['open'] * 2 + df['low'] + df['high']) / 7

    if len(price_trend) < 68:  # Ensure enough data points for the largest span
        print("Insufficient data for trend calculations.")
        return None, None, None, None, None

    # Calculate moving averages
    trend_ma = price_trend.ewm(span=3, adjust=False).mean()
    weight_ma = weight(trend_ma, 6)

    # Calculate trendline
    ema_21 = price_trend.ewm(span=21, adjust=False).mean()
    ema_34 = price_trend.ewm(span=34, adjust=False).mean()
    ema_68 = price_trend.ewm(span=68, adjust=False).mean()
    trendline = (ema_21 + ema_34 + ema_68) / 3

    # Calculate plot_trendline
    plot_trendline = weight(trendline, 6)

    paletteT = ['lime' if trendline.iloc[i] <= trendline.iloc[i - 1] else 'purple' for i in range(1, len(df))]
    paletteT.insert(0, 'lime')  # Default for the first value
    palette = ['red' if weight_ma.iloc[i] >= weight_ma.iloc[i - 1] else 'lime' for i in range(1, len(weight_ma))]
    palette.insert(0, 'red')  # Default for the first value

    # Calculate higher and lower values for candlesticks
    higher = np.where(weight_ma >= weight_ma.shift(1), weight_ma, weight_ma.shift(1))
    lower = np.where(weight_ma < weight_ma.shift(1), weight_ma, weight_ma.shift(1))

    return higher, lower, palette, plot_trendline, paletteT


# Function to display chart
def display_chart(stock_symbol):
    df = load_stock_data(stock_symbol)
    timeframe = st.session_state.get("period", "Daily")

    if df is None:
        st.error(f"No data available for the stock symbol '{stock_symbol}'. Please check the symbol or the data file.")
        return

    try:
        # Ensure 'datetime' is a pandas datetime type
        df['date'] = pd.to_datetime(df.index)

        if timeframe == "Weekly":
            # Filter to keep only the last 2 years
            df = df.tail(96)
        else:
            # Filter to keep only the last 120 rows - 6 months
            df = df.tail(120)

        # Create a full range of dates from the minimum to the maximum date in your data
        all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())

        # Find missing dates
        missing_dates = all_dates.difference(df['date'])
        higher, lower, palette, plot_trendline, paletteT = calculate_trend_data(df)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.2, 0.3],
            vertical_spacing=0.03,
        )

        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Add volume trace
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['volume'],
                name="Volume",
                marker_color='blue',
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

        #Add 趋势专家 trendline
        fig.add_trace(
             go.Scatter(
                 x=df['date'],
                 y=plot_trendline,
                 mode='markers',
                 marker=dict(color=paletteT, size=4),  # make dots for color change
                 name="Trendline",
            ),
            row=3,
            col=1,
        )

        for i in range(len(df)):
            fig.add_trace(
                go.Scatter(
                    x=[df['date'].iloc[i], df['date'].iloc[i]],
                    y=[lower[i], higher[i]],
                    mode='lines',
                    line=dict(color=palette[i], width=2),
                    showlegend=False
                ),
                row=3,
                col=1,
            )

        fig.update_xaxes(
            rangebreaks=[
                dict(values=missing_dates),  # Skip missing dates
            ]
        )

        if timeframe == "Weekly":
            title = f"两年走势图 - {stock_symbol} ({timeframe} Chart)"  # 2 years for Weekly
        else:
            title = f"六个月走势图 - {stock_symbol} ({timeframe} Chart)"  # 6 months for Daily

        fig.update_layout(
            title= title,
            xaxis=dict(
                title=None,
                showgrid=False,
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(title="股价 Price", side="left"),
            yaxis2=dict(title="交易量 Volume", side="left"),
            yaxis3=dict(title="趋势专家 Trend", side="left"),
            height=800,
            showlegend=False,
        )

        # Display the static plotly chart in Streamlit
        st.sidebar.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': False,  # Disable the mode bar entirely
            'staticPlot': True  # Make the chart static and non-interactive
        })

    except Exception as e:
        st.error(f"An error occurred while creating the chart: {e}")

def logout_user():
    """Reset session state and log the user out."""
    for key in ['logged_in', 'otp_sent', 'verified', 'user_id','show_list']:
        st.session_state[key] = False  # Reset all login-related states
    st.session_state['email'] = ""  # Clear the email input
    st.session_state['criteria'] = {}
    st.session_state['selected_stock'] = None
    st.session_state['matching_stocks'] = []

def main():
    st.title("选股平台 Stock Screener")
    update = get_latest_date()
    # Display latest date


    add_custom_css()
    # Initialize the number of matching stocks
    temp_file_path = None

    if 'logged_in' not in st.session_state and 'otp_sent' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['email'] = ""
        st.session_state['selected_stock'] = None
        st.session_state['show_list'] = False
        st.session_state['criteria'] = {}
        st.session_state['matching_stocks'] = []
        st.session_state['temp_file_path'] = None
        st.session_state['user_id'] = None
        st.session_state['otp_sent'] = False  # Track if OTP has been sent
        st.session_state["verified"] = False
        st.session_state['selected_exchange'] = "MYX"
        st.session_state['period'] = "Daily"

    if not st.session_state["logged_in"]:
        st.subheader("登入 Login")

        # Input for user email
        email_user = st.text_input("电邮 Email", key="email_input")
        email = email_user.strip().lower()
        # step 1: send OTP handling
        if not st.session_state["otp_sent"]:
            if st.button("发送密码 Send OTP"):
                # Send OTP to user's email
                try:
                    user = auth.get_user_by_email(email)  # Check if user exists
                    user_id = email  # Get the user ID from Firebase
                    send_otp(email)  # Call the function to generate OTP
                    st.session_state["otp_sent"] = True
                    st.session_state["user_id"] = user_id
                    st.button("OK")
                except firebase_admin.auth.UserNotFoundError:
                    st.error("电邮输入错误，请再尝试。Email not found. Please try again. ")
                except Exception as e:
                    st.error(f"Error sending OTP: {e}")

        # Step 2: Verify OTP
        elif not st.session_state["verified"]:
            otp = st.text_input("输入密码 Enter OTP", type="password")
            if st.button("验证密码 Verify OTP"):
                if validate_otp(st.session_state["user_id"], otp):
                    st.session_state["verified"] = True
                    st.success("验证成功！请按 ‘进入’ 键。 OTP verified successfully! Click 'Enter App'.")
                    st.button("进入 Enter App", on_click=lambda: st.session_state.update({"logged_in": True}))
                else:
                    st.error("密码错误/逾期,请再领取新密码。 Invalid or expired OTP. Please request a new OTP.")
                    # Reset the OTP flow to allow retry
                    st.button("重新发送密码 Resend new OTP")
                    st.session_state["otp_sent"] = False
                    st.session_state["user_id"] = None

    else:
        st.sidebar.button("登出 Logout", on_click=logout_user)

        # Store the previous exchange selection
        previous_exchange = st.session_state['selected_exchange']
        st.markdown(f"### 📈 所选股市：   Select Exchange：")
        exchange = st.selectbox("", ["MYX", "NASDAQ", "NYSE", "HKEX", "SSE", "SZSE","SGX"])
        st.write(" ")
        st.write(" ")
        previous_timeframe = st.session_state['period']
        st.markdown("### 🕒 图表周期 (Chart Interval)")
        timeframe = st.selectbox("", ["Daily", "Weekly"])
        st.session_state['period'] = timeframe

        # If user changes exchange, reset session state and refresh page
        if exchange != previous_exchange or timeframe != previous_timeframe:
            # Reset only relevant variables while keeping other session data
            st.session_state['selected_exchange'] = exchange  # Store new exchange
            st.session_state['selected_stock'] = None
            st.session_state['show_list'] = False
            st.session_state['matching_stocks'] = []

        update = get_latest_date()
        st.markdown(f"### 📅 {exchange}-{timeframe} 数据最后更新 (Data Last Update): {update}")

        st.write(" ")
        st.write(" ")

        # Checkboxes for indicators
        st.markdown(
            "<h5>⭐ <span style='color: #FF4500; font-size: 30px;'>选股条件 Stock Screening Criteria</span> ⭐</h5>",
            unsafe_allow_html=True
        )
        st.write("股票必须满足所有条件： Stocks must meet all the selected criteria:")
        st.write("")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "<h5>🔷 <span style='color: #1E90FF; font-size: 20px;'>五道彩图均线选项</span> 🔷</h5>",
                unsafe_allow_html=True
            )
            r1_selected = st.checkbox("均线： 10日 > 20日 > 30日", key="r1_check",
                                      value=st.session_state['criteria'].get('r1', False))
            r2_selected = st.checkbox("均线： 20日 > 30日 > 60日", key="r2_check",
                                      value=st.session_state['criteria'].get('r2', False))
            r3_selected = st.checkbox("均线： 60日 > 120日 > 240日", key="r3_check",
                                      value=st.session_state['criteria'].get('r3', False))


            st.write(" ")
            st.write(" ")
            st.markdown(
                "<h5>🔷 <span style='color: #1E90FF; font-size: 20px;'>趋势专家</span> 🔷</h5>",
                unsafe_allow_html=True
            )
            qs_selected = st.checkbox("红柱紫线", key="qs_rbpl_check",
                                      value=st.session_state['criteria'].get('qsrbpl', False))
            qsgrb_selected = st.checkbox("绿变红", key="qs_grb_check",
                                         value=st.session_state['criteria'].get('qsgrb', False))
            qsrgb_selected = st.checkbox("红变绿", key="qs_rgb_check",
                                         value=st.session_state['criteria'].get('qsrgb', False))
            qsgpl_selected = st.checkbox("趋势线绿变紫", key="qs_gpl_check",
                                         value=st.session_state['criteria'].get('qsgpl', False))
            qspgl_selected = st.checkbox("趋势线紫变绿", key="qs_pgl_check",
                                         value=st.session_state['criteria'].get('qspgl', False))
            qsatm_selected = st.checkbox("ATM （宽松版）", key="qs_atm_check",
                                         value=st.session_state['criteria'].get('qsatm', False))
            qsatmS_selected = st.checkbox("ATM （严谨版）", key="qs_atmS_check",
                                         value=st.session_state['criteria'].get('qsatmS', False))
            qsbar_selected = st.selectbox(
                "连续红柱天数:",
                options=[0, 2, 3, 4, 5],
                index=[0, 2, 3, 4, 5].index(st.session_state['criteria'].get('qsrbd', 0)),
                key="qs_rbd_check"
            )

            st.write(" ")
            st.write(" ")
            st.markdown(
                "<h5>🔷 <span style='color: #1E90FF; font-size: 20px;'>多空王</span> 🔷</h5>",
                unsafe_allow_html=True
            )
            DKWR_selected = st.checkbox("红飘带", key="DKWR_check", value=st.session_state['criteria'].get('DKWR', False))
            DKWB_selected = st.checkbox("蓝飘带", key="DKWB_check", value=st.session_state['criteria'].get('DKWB', False))
            R2B_selected = st.checkbox("飘带红变蓝", key="R2B_check", value=st.session_state['criteria'].get('DKWR2B', False))
            B2R_selected = st.checkbox("飘带蓝变红", key="B2R_check", value=st.session_state['criteria'].get('DKWB2R', False))
            DKWRD_selected = st.selectbox(
                "连续红飘带天数:",
                options=[0, 2, 3, 4, 5],
                index=[0, 2, 3, 4, 5].index(st.session_state['criteria'].get('DKWRD', 0)),
                key="DKWRD_check"
            )
            DKWBD_selected = st.selectbox(
                "连续蓝飘带天数:",
                options=[0, 2, 3, 4, 5],
                index=[0, 2, 3, 4, 5].index(st.session_state['criteria'].get('DKWBD', 0)),
                key="DKWBD_check"
            )

            st.write(" ")
            st.write(" ")
            st.markdown(
                "<h5>🔷 <span style='color: #1E90FF; font-size: 20px;'>主力资金</span> 🔷</h5>",
                unsafe_allow_html=True
            )
            ZLB0_selected = st.checkbox("突破零轴线", key="ZLB0_check", value=st.session_state['criteria'].get('ZLB0', False))
            ZLR_selected = st.checkbox("红色", key="ZLR_check",
                                        value=st.session_state['criteria'].get('ZLR', False))
            ZLG2R_selected = st.checkbox("绿变红", key="ZLG2R_check", value=st.session_state['criteria'].get('ZLG2R', False))
            ZLR2G_selected = st.checkbox("红变绿", key="ZLR2G_check", value=st.session_state['criteria'].get('ZLR2G', False))


        with col2:

            st.markdown(
                "<h5>🔷 <span style='color: #1E90FF; font-size: 20px;'>资金所向</span> 🔷</h5>",
                unsafe_allow_html=True
            )
            zj_selected = st.checkbox("水上红柱", key="zj_check", value=st.session_state['criteria'].get('zj', False))
            zjg2r_selected = st.checkbox("绿变红", key="zj_G2R_check",
                                         value=st.session_state['criteria'].get('zjg2r', False))
            zjr2g_selected = st.checkbox("红变绿", key="zj_R2G_check",
                                         value=st.session_state['criteria'].get('zjr2g', False))
            zjbar_selected = st.selectbox(
                "连续红柱天数:",
                options=[0, 2, 3, 4, 5],
                index=[0, 2, 3, 4, 5].index(st.session_state['criteria'].get('zjrbd', 0)),
                key="zj_rbd_check"
            )

            st.write(" ")
            st.write(" ")
            st.markdown(
                "<h5>🔷 <span style='color: #1E90FF; font-size: 20px;'>资金图</span> 🔷</h5>",
                unsafe_allow_html=True
            )
            brsiMma_selected = st.checkbox("主力上穿均线", key="brsiMma_check",
                                           value=st.session_state['criteria'].get('brsiMma', False))
            brsi1Mma_selected = st.checkbox("主力首次上穿均线", key="brsi1Mma_check",
                                            value=st.session_state['criteria'].get('brsi1Mma', False))
            rrsi1_selected = st.checkbox("散户首次归零", key="rrsi1_check",
                                            value=st.session_state['criteria'].get('rrsi1', False))

            # st.write("主力1： ")
            col1, col2 = st.columns([0.5, 1])  # Adjust width as needed
            with col1:
                brsi_operator_selected = st.selectbox(" 主力1：", options=[0, ">=", "<=", "="],
                                                      index=[0, ">=", "<=", "="].index(st.session_state['criteria'].get('brsio', 0)), key="brsi_operator")
            with col2:
                brsi_value = st.number_input(" ", min_value=0.0, max_value=100.0, key="brsi_value",
                                             value=float(st.session_state['criteria'].get('brsi_value', 0.0)), step=0.1,
                                             disabled=(brsi_operator_selected == 0))
            if brsi_operator_selected == 0:
                brsi_value = 0.0  # Reset stored value

            # st.write("主力2： ")
            col1, col2 = st.columns([0.5, 1])  # Adjust width as needed
            with col1:
                brsi2_operator_selected = st.selectbox(" 主力2：", options=[0, ">=", "<=", "="],
                                                      index=[0, ">=", "<=", "="].index(st.session_state['criteria'].get('brsio2', 0)), key="brsi2_operator")
            with col2:
                brsi2_value = st.number_input(" ", min_value=0.0, max_value=100.0, key="brsi2_value",
                                             value=float(st.session_state['criteria'].get('brsi2_value', 0.0)), step=0.1,
                                             disabled=(brsi2_operator_selected == 0))
            if brsi2_operator_selected == 0:
                brsi2_value = 0.0  # Reset stored value

            # st.write("游资： ")
            col1, col2 = st.columns([0.5, 1])  # Adjust width as needed
            with col1:
                hrsi_operator_selected = st.selectbox(" 游资：", options=[0, ">=", "<=", "="],
                                                      index=[0, ">=", "<=", "="].index(st.session_state['criteria'].get('hrsio', 0)), key="hrsi_operator")
            with col2:
                hrsi_value = st.number_input(" ", min_value=0.0, max_value=100.0, key="hrsi_value",
                                             value=float(st.session_state['criteria'].get('hrsi_value', 0.0)), step=0.1,
                                             disabled=(hrsi_operator_selected == 0))
            if hrsi_operator_selected == 0:
                hrsi_value = 0.0  # Reset stored value

            # st.write("散户： ")
            col1, col2 = st.columns([0.5, 1])  # Adjust width as needed
            with col1:
                rrsi_operator_selected = st.selectbox(" 散户：", options=[0, ">=", "<=", "="],
                                                      index=[0, ">=", "<=", "="].index(st.session_state['criteria'].get('rrsio', 0)),key="rrsi_operator")
            with col2:
                rrsi_value = st.number_input(" ", min_value=0.0, max_value=100.0, key="rrsi_value",
                                             value=float(st.session_state['criteria'].get('rrsi_value', 0.0)), step=0.1,
                                             disabled=(rrsi_operator_selected == 0))
            if rrsi_operator_selected == 0:
                rrsi_value = 0.0  # Reset stored value

            st.write(" ")
            st.write(" ")
            st.markdown(
                "<h5>🔷 <span style='color: #1E90FF; font-size: 20px;'>RSI</span> 🔷</h5>",
                unsafe_allow_html=True
            )
            rsi_selected = st.checkbox("RSI", key="rsi_check", value=st.session_state['criteria'].get('rsi', False))
            # Use columns to arrange inputs in one row
            col1, col2, = st.columns([1, 1])  # Adjust width as needed
            with col1:
                # RSI Min Value Input (Disabled if checkbox is unchecked)
                rsi_min = st.number_input(
                    "Min RSI:", min_value=0, max_value=100,
                    value=st.session_state['criteria'].get('rsi_min', 0), step=1,
                    disabled=not rsi_selected
                )

            with col2:
                # RSI Max Value Input (Disabled if checkbox is unchecked)
                rsi_max = st.number_input(
                    "Max RSI:", min_value=0, max_value=100,
                    value=st.session_state['criteria'].get('rsi_max', 100), step=1,
                    disabled=not rsi_selected
                )

            if not rsi_selected:
                rsi_min = 0
                rsi_max = 0

            st.write(" ")
            st.write(" ")
            st.markdown(
                "<h5>🔷 <span style='color: #1E90FF; font-size: 20px;'>神奇九转</span> 🔷</h5>",
                unsafe_allow_html=True
            )
            n1_selected = st.checkbox("牛一", key="n1_check", value=st.session_state['criteria'].get('n1', False))
            x9_selected = st.checkbox("熊九", key="x9_check", value=st.session_state['criteria'].get('x9', False))

            st.write(" ")
            st.write(" ")
            st.markdown(
                "<h5>🔷 <span style='color: #1E90FF; font-size: 20px;'>MACD</span> 🔷</h5>",
                unsafe_allow_html=True
            )
            gc_selected = st.checkbox("金叉", key="gc_check", value=st.session_state['criteria'].get('gc', False))
            dc_selected = st.checkbox("死叉", key="dc_check", value=st.session_state['criteria'].get('dc', False))
            a0_selected = st.checkbox("零轴上", key="a0_check", value=st.session_state['criteria'].get('a0', False))
            b0_selected = st.checkbox("零轴下", key="b0_check", value=st.session_state['criteria'].get('b0', False))

            st.write(" ")
            st.write(" ")
            st.markdown(
                "<h5>🔷 <span style='color: #1E90FF; font-size: 20px;'>其它选股指标</span> 🔷</h5>",
                unsafe_allow_html=True
            )
            y1_selected = st.checkbox("第一黄柱", key="y1_check", value=st.session_state['criteria'].get('y1', False))
            dxw_selected = st.checkbox("短线王", key="dxw_check", value=st.session_state['criteria'].get('dxw', False))
            vm_selected = st.checkbox("交易量大于昨日", key="vm_check", value=st.session_state['criteria'].get('vm', False))
            vm2_selected = st.checkbox("倍量", key="vm2_check", value=st.session_state['criteria'].get('vm2', False))
            pmc_selected = st.checkbox("收市价高于昨日收市价", key="pmc_check", value=st.session_state['criteria'].get('pmc', False))
            pmh_selected = st.checkbox("收市价高于昨日最高价", key="pmh_check",
                                       value=st.session_state['criteria'].get('pmh', False))

        st.write(" ")
        st.write(" ")
        min_volume = st.number_input("最低成交量为 100,000 的倍数:   Minimum Volume as a multiple of 100,000:",
                                     min_value=0,
                                     value=st.session_state['criteria'].get('min_volume', 0), step=1)
        min_price = st.number_input("最低股价:   Minimum Stock Price:", min_value=0.0,
                                    value=st.session_state['criteria'].get('min_price', 0.0), step=0.1)

        # Update criteria in session state
        st.session_state['criteria'] = {
            'r1': r1_selected,
            'r2': r2_selected,
            'r3': r3_selected,
            'n1': n1_selected,
            'x9': x9_selected,
            'y1': y1_selected,
            'dxw': dxw_selected,
            'zj': zj_selected,
            'vm': vm_selected,
            'vm2': vm2_selected,
            'pmc': pmc_selected,
            'pmh': pmh_selected,
            'zjrbd': zjbar_selected,
            'zjg2r': zjg2r_selected,
            'zjr2g': zjr2g_selected,
            'qsrbpl': qs_selected,
            'qsrbd': qsbar_selected,
            'qsgrb': qsgrb_selected,
            'qsrgb': qsrgb_selected,
            'qsgpl': qsgpl_selected,
            'qspgl': qspgl_selected,
            'qsatm': qsatm_selected,
            'qsatmS': qsatmS_selected,
            'DKWR': DKWR_selected,
            'DKWB': DKWB_selected,
            'DKWR2B': R2B_selected,
            'DKWB2R': B2R_selected,
            'DKWRD':DKWRD_selected,
            'DKWBD':DKWBD_selected,
            'ZLB0': ZLB0_selected,
            'ZLR': ZLR_selected,
            'ZLG2R': ZLG2R_selected,
            'ZLR2G': ZLR2G_selected,
            'dc':dc_selected,
            'gc':gc_selected,
            'a0':a0_selected,
            'b0':b0_selected,
            'min_volume': min_volume,
            'min_price': min_price,
            'rsi': rsi_selected,
            'rsi_min': rsi_min,
            'rsi_max': rsi_max,
            'brsiMma': brsiMma_selected,
            'brsi1Mma': brsi1Mma_selected,
            'rrsi1': rrsi1_selected,
            'brsio':brsi_operator_selected,
            'brsi_value': brsi_value,
            'brsio2': brsi2_operator_selected,
            'brsi2_value': brsi2_value,
            'hrsio': hrsi_operator_selected,
            'hrsi_value': hrsi_value,
            'rrsio': rrsi_operator_selected,
            'rrsi_value': rrsi_value
        }

        col5, col6 = st.columns([1, 1])

        with col5:
            if st.button("选股 Screen Stock"):
                st.session_state['show_list'] = False
                try:
                    # Authenticate and download the file content (returns a DataFrame)
                    file_content = download_file()

                    if isinstance(file_content, pd.DataFrame) and not file_content.empty:
                        matching_symbols, temp_file_path = check_indicators_and_save(
                            file_content, min_volume, min_price, brsi_value, brsi2_value, hrsi_value, rrsi_value, rsi_min, rsi_max)
                        st.session_state['temp_file_path'] = temp_file_path
                        st.session_state['matching_stocks'] = matching_symbols

                        if matching_symbols:
                            st.session_state['show_list'] = True
                            # st.success("Matching stocks are found!")
                        else:
                            st.warning("No stocks found matching all selected criteria.")
                    else:
                        st.error("Error: The downloaded file content is empty or invalid.")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")

        # if exchange != st.session_state.selected_exchange: #detect changes in exchange
        #     st.session_state.selected_exchange = exchange
        #     st.session_state.matching_stocks = []  # Clear stock list

        # Display matching stocks
        if st.session_state.get('show_list', False):
            stock_list = st.session_state.get('matching_stocks', [])
            if stock_list:
                st.write(f"符合条件的股数量  Number of stocks meeting the criteria: {len(stock_list)}")
                display_symbols_dropdown(stock_list)

        # Download button
        with col6:
            if st.session_state.get('temp_file_path'):
                today_date = datetime.now().strftime('%d%m%y')
                try:
                    # Read the file content
                    with open(st.session_state['temp_file_path'], "r") as file:
                        file_data = file.readlines()

                    # Prefix each stock symbol with "MYX:"
                    selected_exchange = st.session_state.get("selected_exchange", "MYX")  # Default to MYX if not set
                    timeframe = st.session_state.get("period", "Daily")
                    modified_data = "\n".join([f"{selected_exchange}:{line.strip()}" for line in file_data])

                    # Streamlit download button
                    if modified_data:
                        st.download_button(
                            label="下载名单 Download List",
                            data=modified_data,
                            file_name=f"filtered_result_{selected_exchange}-{timeframe}_{today_date}.txt",
                            mime="text/plain"
                        )
                except FileNotFoundError:
                    st.warning("No file available to download.")

        # Define a mapping for display labels
        criteria_labels = {
            "r1": "五道彩图均线： 10日 > 20日 > 30日",
            "r2": "五道彩图均线： 20日 > 30日 > 60日",
            "r3": "五道彩图均线： 60日 > 120日 > 240日",
            "n1": "神奇九转 - 牛一",
            "x9": "神奇九转 - 熊九",
            "y1": "第一黄柱",
            "dxw": "短线王",
            "vm": "交易量大于昨日",
            "vm2": "倍量",
            "pmc": "收市价高于昨日收市价",
            "pmh": "收市价高于昨日最高价",
            "dc": "MACD - 死叉",
            "gc": "MACD - 金叉",
            "a0": "MACD - 零轴上",
            "b0": "MACD - 零轴下",
            "zj": "资金所向 - 水上红柱",
            "zjg2r": "资金所向 - 绿变红",
            "zjr2g": "资金所向 - 红变绿",
            "zjrbd": "资金所向 - 连续红柱天数",
            "qsrbpl": "趋势专家 - 红柱紫线",
            "qsgrb": "趋势专家 - 绿变红",
            "qsrgb": "趋势专家 - 红变绿",
            "qsgpl": "趋势专家 - 趋势线绿变紫",
            "qspgl": "趋势专家 - 趋势线紫变绿",
            "qsrbd": "趋势专家 - 连续红柱天数",
            "qsatm": "趋势专家 - ATM (宽松版）",
            "qsatmS": "趋势专家 - ATM (严谨版）",
            "DKWR": "多空王 - 红飘带",
            "DKWB": "多空王 - 蓝飘带",
            "DKWR2B": "多空王 - 飘带红变蓝",
            "DKWB2R": "多空王 - 飘带蓝变红",
            "DKWRD": "多空王 - 连续红飘带天数",
            "DKWBD": "多空王 - 连续蓝飘带天数",
            "ZLB0": "主力资金 - 突破零轴线",
            "ZLG2R": "主力资金 - 绿变红",
            "ZLR2G": "主力资金 - 红变绿",
            "ZLR": "主力资金 - 红色",
            "min_volume": "最低成交量（100k的倍数）",
            "min_price": "最低股价",
            "rsi": "RSI",
            "rsi_min": "最低 RSI 参数",
            "rsi_max": "最高 RSI 参数",
            "brsiMma": "资金图 - 筹码上穿均线",
            "brsi1Mma": "资金图 - 筹码首次上穿均线",
            "rrsi1": "资金图 - 散户首次归零",
            "brsio": "资金图 - 主力参数关系",
            "brsi_value": "资金图 - 主力参数",
            "brsio2": "资金图 - 主力参数关系2",
            "brsi2_value": "资金图 - 主力参数2",
            "hrsio": "资金图 - 游资参数关系",
            "hrsi_value": "资金图 - 游资参数",
            "rrsio": "资金图 - 散户参数关系",
            "rrsi_value": "资金图 - 散户参数",
        }

        # Display selected criteria in the sidebar without extra spacing
        st.sidebar.markdown("""
            <h3 style='color:blue; font-family:sans-serif; text-decoration:underline;'>已选条件 Selected Criteria</h3>
        """, unsafe_allow_html=True)
        criteria = st.session_state.get('criteria', {})
        if criteria:
            selected_criteria = "".join(
                f"<div style='line-height:1.2'><b>{criteria_labels.get(key, key.replace('_', ' ').capitalize())}:</b> {value}</div>"
                for key, value in criteria.items() if value
            )
            st.sidebar.markdown(selected_criteria, unsafe_allow_html=True)

        # Display selected stock chart
        if st.session_state['selected_stock']:
            display_stock = st.session_state['selected_stock'].replace(f'{selected_exchange}:', '')
            display_chart(display_stock)


if __name__ == "__main__":
    main()
