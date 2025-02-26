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


if not firebase_admin._apps:
    firebase_creds = dict(st.secrets["firebase_credentials"])
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

db = firestore.client()
print("Firebase initialized successfully!")

def get_latest_date():
    """Fetch the last updated date from Firestore (only once)."""
    doc_ref = db.collection("data_date").document("latest_date")
    doc = doc_ref.get()

    if doc.exists:
      #  print(f"Last recorded date: {doc.to_dict().get('date')}")
        return doc.to_dict().get('date')
    else:
        print("No date found.")

def send_otp(email):
    """Generate and store OTP, then send it to the user's email."""
    user_ref = db.collection("users").document(email)
    user = user_ref.get()

    if user.exists:
        otp = str(random.randint(100000, 999999))  # Generate a 6-digit OTP
        expiration_time = datetime.now() + timedelta(minutes=5)  # OTP valid for 5 minutes

        # ✅ Store OTP with email as document ID
        otp_ref = db.collection("otp_verifications").document(email)
        otp_ref.set({
            "otp": otp,
            "expiration_time": expiration_time.isoformat(),
            "verified": False
        })

        #st.success("OTP generated! Please check Telegram to receive it.")
    else:
        st.error("Email not found in Firestore. Please register first.")

def validate_otp(email, otp):
    """Validate the OTP provided by the user."""
    try:
        otp_doc = db.collection("otp_verifications").document(email).get()
        if not otp_doc.exists:
            print("No OTP request found for this user.")
            return False

        otp_data = otp_doc.to_dict()
        stored_otp = otp_data["otp"]
        expiration_time = datetime.fromisoformat(otp_data["expiration_time"])

        # Check if OTP matches and has not expired
        if datetime.now() > expiration_time:
            print("OTP has expired.密码逾期")
            return False

        if otp == stored_otp:
            # Mark the OTP as verified
            db.collection("otp_verifications").document(email).update({"verified": True})
            print("OTP validated successfully!成功验证！")
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
            background-color: #333333; 
            color: white;
            font-weight: bold;

        }
        div.stDownloadButton > button:hover {
            background-color: #555555; 
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

def download_file():
    """Retrieve all stock data from the `screened_result` collection and return as a DataFrame."""
    collection_ref = db.collection("screened_result")
    docs = collection_ref.stream()

    stock_list = []
    for doc in docs:
        stock_data = doc.to_dict()
        stock_list.append(stock_data)

    if not stock_list:
        print("No data found in Firestore collection: screened_result.")
        return None

    # Convert list of stock data to Pandas DataFrame
    df = pd.DataFrame(stock_list)
    return df

def load_stock_data(stock_symbol):
    """Load stock data from Firestore and return a Pandas DataFrame."""
    doc_ref = db.collection("stocks").document(stock_symbol)  # Reference to Firestore document
    doc = doc_ref.get()  # Retrieve document

    if doc.exists:
        stock_data = doc.to_dict().get("data", [])  # Get the 'data' list
        if stock_data:
            df = pd.DataFrame(stock_data)  # Convert list of dictionaries to DataFrame
            df["date"] = pd.to_datetime(df["date"])  # Ensure date is in datetime format
            df = df.set_index("date")  # Set 'date' as the index
            return df
        else:
            st.error(f"No data available for {stock_symbol}.")
            return None
    else:
        st.error(f"Stock {stock_symbol} not found in Firestore.")
        return None

def display_symbols_dropdown(symbols):
    selected_stock = st.selectbox("请选择一只股票查看图表： Select a stock to view the chart:", options=symbols)
    if st.button("看图表 Show Chart"):
        select_stock(selected_stock)
        st.session_state['selected_stock'] = selected_stock

# Function to handle stock selection
def select_stock(stock):
    st.session_state['selected_stock'] = stock


def check_indicators_and_save(df, min_volume, min_price, min_banker_value, max_banker_value,rsi_min, rsi_max):
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
        if st.session_state.get('y1_check', False):
            mask &= (df['y1'] == 1)
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
        if st.session_state.get('qs_rbd_check', False):
            qs_selection = st.session_state.get('qs_rbd_check',0)
            if qs_selection == 2:
                mask &= (df['qs2R'] == 1)
            elif qs_selection == 3:
                mask &= (df['qs3R'] == 1)
            elif qs_selection == 4:
                mask &= (df['qs4R'] == 1)
            elif qs_selection == 5:
                mask &= (df['qs5R'] == 1)

        # Apply user-defined filters
        if min_volume:
            mask &= (df['volume'] >= min_volume * 100000)
        if min_price:
            mask &= (df['close'] >= min_price)
        if min_banker_value:
            mask &= (df['brsi'] >= min_banker_value)
        if max_banker_value:
            mask &= (df['brsi'] <= max_banker_value)
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
    if df is None:
        st.error(f"No data available for the stock symbol '{stock_symbol}'. Please check the symbol or the data file.")
        return

    try:
        # Ensure 'datetime' is a pandas datetime type
        df['date'] = pd.to_datetime(df.index)

        # Filter to keep only the last 120 rows
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

        fig.update_layout(
            title=f"六个月走势图 {stock_symbol}",
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
    st.markdown(f"### 📅 数据最后更新 Data Last Update: {update}")

    add_custom_css()
    # Initialize the number of matching stocks
    temp_file_path = None

    if 'logged_in' and 'otp_sent' not in st.session_state:
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

    if not st.session_state["logged_in"]:
        st.subheader("登入 Login")

        # Input for user email
        email = st.text_input("电邮 Email", key="email_input")

        # step 1: send OTP handling
        if not st.session_state["otp_sent"]:
            if st.button("发送密码 Send OTP"):
                # Send OTP to user's email
                try:
                    user = auth.get_user_by_email(email)  # Check if user exists
                    user_id = email  # Get the user ID from Firebase
                    send_otp(email)  # Call the function to send OTP
                    st.session_state["otp_sent"] = True
                    st.session_state["user_id"] = user_id
                    st.success("密码已经发出。请到 Telegram 输入 /otp 领取密码。")
                    st.success("OTP sent to your telegram. Please type /otp in your Telegram.")
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
        # Checkboxes for indicators
        st.write("选股条件（股票必须满足所有条件）：")
        st.write("Select indicators (stocks must meet all selected criteria):")
        col1, col2 = st.columns(2)

        with col1:
            st.write("*** 彩图均线选项 ***")
            r1_selected = st.checkbox("彩图均线： 5日 > 10日 > 20日", key="r1_check",
                                           value=st.session_state['criteria'].get('r1', False))
            r2_selected = st.checkbox("彩图均线： 20日 > 30日 > 60日", key="r2_check",
                                           value=st.session_state['criteria'].get('r2', False))
            r3_selected = st.checkbox("彩图均线： 60日 > 120日 > 240日", key="r3_check",
                                           value=st.session_state['criteria'].get('r3', False))
            st.write("")
            n1_selected = st.checkbox("牛一", key="n1_check", value=st.session_state['criteria'].get('n1', False))
            y1_selected = st.checkbox("第一黄柱", key="y1_check", value=st.session_state['criteria'].get('y1', False))

        with col2:
            st.write("*** 趋势专家 ***")
            qs_selected = st.checkbox("红柱紫线", key="qs_rbpl_check", value=st.session_state['criteria'].get('qsrbpl', False))
            qsgrb_selected = st.checkbox("柱线绿变红", key="qs_grb_check",value=st.session_state['criteria'].get('qsgrb', False))
            qsrgb_selected = st.checkbox("柱线红变绿", key="qs_rgb_check",value=st.session_state['criteria'].get('qsrgb', False))
            qsgpl_selected = st.checkbox("主线绿变紫", key="qs_gpl_check",value=st.session_state['criteria'].get('qsgpl', False))
            qspgl_selected = st.checkbox("主线紫变绿", key="qs_pgl_check",value=st.session_state['criteria'].get('qspgl', False))
            qsbar_selected = st.selectbox(
                "连续红柱天数",
                options=[0, 2, 3, 4, 5],
                key="qs_rbd_check"
            )

            st.write("")
            zj_selected = st.checkbox("资金所向", key="zj_check", value=st.session_state['criteria'].get('zj', False))
            rsi_selected = st.checkbox("RSI", key="rsi_check", value=st.session_state['criteria'].get('rsi', False))
            # Use columns to arrange inputs in one row
            col1, col2, col3 = st.columns([0.5, 1, 0.5])  # Adjust width as needed
            with col1:
                # RSI Min Value Input (Disabled if checkbox is unchecked)
                rsi_min = st.number_input(
                    "Min RSI", min_value=0, max_value=100,
                    value=st.session_state['criteria'].get('rsi_min', 0), step=1,
                    disabled=not rsi_selected,
                    label_visibility="collapsed"
                )

            with col2:
                st.markdown("<h5 style='text-align: center;'>&lt; RSI range &lt;</h5>", unsafe_allow_html=True)

            with col3:
                # RSI Max Value Input (Disabled if checkbox is unchecked)
                rsi_max = st.number_input(
                    "Max RSI", min_value=0, max_value=100,
                    value=st.session_state['criteria'].get('rsi_max', 0), step=1,
                    disabled=not rsi_selected,
                    label_visibility="collapsed"
                )

            if not rsi_selected:
                rsi_min = 0
                rsi_max = 0

        # Input fields for filters
        min_volume = st.number_input("最低成交量为 100,000 的倍数   Minimum Volume as a multiple of 100,000",
                                     min_value=0,
                                     value=st.session_state['criteria'].get('min_volume', 0), step=1)
        min_price = st.number_input("最低股价   Minimum Stock Price", min_value=0.0,
                                    value=st.session_state['criteria'].get('min_price', 0.0), step=0.1)
        min_banker_value = st.number_input("最低红柱   Minimum Banker Value (0-100)", min_value=0,
                                           value=st.session_state['criteria'].get('min_banker_value', 0), step=1)
        max_banker_value = st.number_input("最高红柱   Maximum Banker Value (0-100)", min_value=0,
                                           value=st.session_state['criteria'].get('max_banker_value', 0), step=1)

        # Update criteria in session state
        st.session_state['criteria'] = {
            'r1': r1_selected,
            'r2': r2_selected,
            'r3': r3_selected,
            'n1': n1_selected,
            'y1': y1_selected,
            'zj': zj_selected,
            'qsrbpl': qs_selected,
            'qsrbd' : qsbar_selected,
            'qsgrb': qsgrb_selected,
            'qsrgb': qsrgb_selected,
            'qsgpl': qsgpl_selected,
            'qspgl': qspgl_selected,
            'min_volume': min_volume,
            'min_price': min_price,
            'min_banker_value': min_banker_value,
            'max_banker_value': max_banker_value,
            'rsi': rsi_selected,
            'rsi_min': rsi_min,
            'rsi_max': rsi_max,
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
                            file_content, min_volume, min_price, min_banker_value, max_banker_value, rsi_min, rsi_max)

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
                    modified_data = "\n".join([f"MYX:{line.strip()}" for line in file_data])

                    # Streamlit download button
                    if modified_data:
                        st.download_button(
                            label="下载名单 Download List",
                            data=modified_data,
                            file_name=f"filtered_result_{today_date}.txt",
                            mime="text/plain"
                        )
                except FileNotFoundError:
                    st.warning("No file available to download.")

        # Manual stock symbol input
        # stock_input = st.text_input("或输入股票代码以查看图表： Or enter a stock symbol for chart viewing:")
        # if stock_input:
        #     select_stock(stock_input.strip())
        #     st.session_state['selected_stock'] = stock_input.strip()

        # Define a mapping for display labels
        criteria_labels = {
            "r1": "彩图均线： 5日 > 10日 > 20日",
            "r2": "彩图均线： 20日 > 30日 > 60日",
            "r3": "彩图均线： 60日 > 120日 > 240日",
            "n1": "牛一",
            "y1": "第一黄柱",
            "zj": "资金所向",
            "qsrbpl": "趋势专家 - 红柱紫线",
            "qsgrb": "趋势专家 - 柱线绿变红",
            "qsrgb": "趋势专家 - 柱线红变绿",
            "qsgpl": "趋势专家 - 主线绿变紫",
            "qspgl": "趋势专家 - 主线紫变绿",
            "qsrbd": "趋势专家 - 连续红柱天数",
            "min_volume": "Min Volume in 100k",
            "min_price": "Min Price",
            "min_banker_value": "Min Banker Value",
            "max_banker_value": "Max Banker Value",
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
            display_stock = st.session_state['selected_stock'].replace('MYX:', '')
            display_chart(display_stock)


if __name__ == "__main__":
    main()
