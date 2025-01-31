import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import tempfile
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, auth,firestore
from datetime import datetime, timedelta
import random

# Google Drive API scope
# SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

folder_id = '1VqBBtvzHOb8FKVgP5r1uoRWEWltPVdeD'

if not firebase_admin._apps:
    firebase_creds = dict(st.secrets["firebase_credentials"])
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred)

db = firestore.client()
print("Firebase initialized successfully!")

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
            background-color: lightblue;
            color: white;
        }
        div.stButton > button:hover {
            background-color: blue;
            color: white;
        }
        </style>

        <style>
        /* Style for the download button */
        div.stDownloadButton > button {
            background-color: #FF6666; 
            color: white;

        }
        div.stDownloadButton > button:hover {
            background-color: darkred; 
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


# Function to authenticate and return the Google API service
def authenticate_drive_api():
    try:
        # Load the credentials from Streamlit secrets (no need for json.loads)
        service_account_info = st.secrets["google_credentials"]

        # Create credentials using the service account info and the specified SCOPES
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )

        # Use the credentials to authorize and build the service
        service = build('drive', 'v3', credentials=credentials)
        # st.text(service)
        return service

    except Exception as e:
        st.error(f"Error during authentication: {e}")
        return None


def list_files_in_folder(service, folder_id):
    """Lists all files in a specified Google Drive folder, handling pagination."""
    try:
        files = []
        page_token = None

        while True:
            # Create the query to list the files in the specified folder
            query = f"'{folder_id}' in parents"

            # Call the Drive API to list the files in the folder
            response = service.files().list(q=query, pageSize=100, fields="nextPageToken, files(id, name)",
                                            pageToken=page_token).execute()

            # Add the files from the current page to the list
            files.extend(response.get('files', []))

            # Check if there is a nextPageToken, which means there are more files to fetch
            page_token = response.get('nextPageToken')
            if not page_token:
                break  # No more pages, stop the loop

        return files

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def download_file(service, file_id):
    """Download a file from Google Drive and return its content as a DataFrame."""
    try:
        # st.text(f"Starting download of file with ID: {file_id}")

        # Request to download the file
        request = service.files().get_media(fileId=file_id)
        file_content = BytesIO()  # Create an in-memory file buffer
        downloader = MediaIoBaseDownload(file_content, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()  # Progress indicator
        # st.text(f"Download progress: {int(status.progress() * 100)}%")

        # After download, check the size of the file
        # st.text(f"Download completed. File size: {file_content.getbuffer().nbytes} bytes.")

        # Ensure we are correctly reading the file content into a DataFrame
        file_content.seek(0)  # Reset the file pointer to the beginning after download
        try:
            # st.text("Attempting to read the file into a DataFrame...")
            df = pd.read_excel(file_content, engine='openpyxl')
            # st.text("File read successfully into DataFrame.")
            return df
        except Exception as e:
            st.text(f"Error reading the Excel file: {e}")
            return None
    except HttpError as error:
        st.text(f"Google Drive API Error: {error}")
        return None
    except Exception as e:
        st.text(f"General error occurred during file download or processing: {e}")
        return None


def load_stock_data(stock_symbol, folder_id):
    service = authenticate_drive_api()
    files = list_files_in_folder(service, folder_id)
    print(f"Found {len(files)} files in the folder.")
    print("Files in folder:", [file['name'] for file in files])
    # Log the stock symbol to see if it's passed correctly
    print(f"Looking for file: {stock_symbol}_div.xlsx")

    if not files:
        return None

    file_to_download = next((file for file in files if file['name'].lower() == f"{stock_symbol.lower()}_div.xlsx"),
                            None)

    if not file_to_download:
        st.error(f"Data file for stock symbol '{stock_symbol}' not found.")
        return None
    return download_file(service, file_to_download['id'])


def display_symbols_dropdown(symbols):
    selected_stock = st.selectbox("请选择一只股票查看图表： Select a stock to view the chart:", options=symbols)
    if st.button("看图表 Show Chart"):
        select_stock(selected_stock)
        st.session_state['selected_stock'] = selected_stock


def display_symbols_in_columns(symbols):
    """
    Display a list of stock symbols in 3 columns and trigger stock selection on button click.

    Args:
        symbols (list): List of stock symbols to display.
    """
    # Number of columns
    num_columns = 2
    # Calculate how many symbols should go in each column
    columns = st.columns([1, 1])

    # Distribute the symbols across the columns
    symbols_per_column = len(symbols) // num_columns
    remaining = len(symbols) % num_columns

    start_index = 0
    for i in range(num_columns):
        end_index = start_index + symbols_per_column + (1 if i < remaining else 0)
        with columns[i]:
            # Iterate through each stock symbol and display it as a button
            for stock in symbols[start_index:end_index]:
                stock = stock.strip()
                if stock:  # Ensure it's not an empty string
                    # Use an on_click for better performance
                    if st.button(stock, key=f"btn_{stock}", on_click=select_stock, args=(stock,)):
                        pass
                        # Call select_stock only if a button is clicked
                        st.session_state['page'] = "Chart Viewer"

        start_index = end_index


# Function to handle stock selection
def select_stock(stock):
    st.session_state['selected_stock'] = stock


def check_indicators_and_save(df, min_volume, min_price, min_banker_value, max_banker_value):
    try:
        # Initialize mask for filtering
        mask = pd.Series([True] * len(df))

        # Apply filters based on checkbox states
        if st.session_state.get('rainbow_check', False):
            mask &= (df['rainbow'] == 1)
        if st.session_state.get('n1_check', False):
            mask &= (df['n1'] == 1)
        if st.session_state.get('y1_check', False):
            mask &= (df['y1'] == 1)
        if st.session_state.get('zj_check', False):
            mask &= (df['zj'] == 1)
        if st.session_state.get('qs_check', False):
            mask &= (df['qs'] == 1)

        # Apply user-defined filters
        if min_volume:
            mask &= (df['volume'] >= min_volume * 100000)
        if min_price:
            mask &= (df['close'] >= min_price)
        if min_banker_value:
            mask &= (df['brsi'] >= min_banker_value)
        if max_banker_value:
            mask &= (df['brsi'] <= max_banker_value)

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
    wma = values.rolling(window=length).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)))
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
    df = load_stock_data(stock_symbol, '1VqBBtvzHOb8FKVgP5r1uoRWEWltPVdeD')
    if df is None:
        st.error(f"No data available for the stock symbol '{stock_symbol}'. Please check the symbol or the data file.")
        return

    try:
        # Ensure 'datetime' is a pandas datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Filter to keep only the last 120 rows
        df = df.tail(120)

        # Create a full range of dates from the minimum to the maximum date in your data
        all_dates = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max())

        # Find missing dates
        missing_dates = all_dates.difference(df['datetime'])
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
                x=df['datetime'],
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
                x=df['datetime'],
                y=df['volume'],
                name="Volume",
                marker_color='blue',
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

        # Add 趋势专家 trendline
        fig.add_trace(
            go.Scatter(
                x=df['datetime'],
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
                    x=[df['datetime'].iloc[i], df['datetime'].iloc[i]],
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
            title=f"六个月走势图 6 Months Chart Viewer {stock_symbol}",
            xaxis=dict(
                title=None,
                showgrid=False,
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(title="股价", side="left"),
            yaxis2=dict(title="交易量", side="left"),
            yaxis3=dict(title="趋势专家", side="left"),
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
    for key in ['logged_in', 'otp_sent', 'verified', 'user_id']:
        st.session_state[key] = False  # Reset all login-related states
    st.session_state['email'] = ""  # Clear the email input

def main():
    st.title("选股平台 Stock Screener")
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
        st.subheader("Login 登入")

        # Input for user email
        email = st.text_input("Email 电邮", key="email_input")

        # step 1: send OTP handling
        if not st.session_state["otp_sent"]:
            if st.button("Send OTP 发送密码"):
                # Send OTP to user's email
                try:
                    user = auth.get_user_by_email(email)  # Check if user exists
                    user_id = email  # Get the user ID from Firebase
                    send_otp(email)  # Call the function to send OTP
                    st.session_state["otp_sent"] = True
                    st.session_state["user_id"] = user_id
                    st.success("OTP sent to your telegram. Please type /otp in your Telegram.")
                    st.success("密码已经发出。请到 Telegram 输入 /otp 领取密码。")
                    st.button("OK")
                except firebase_admin.auth.UserNotFoundError:
                    st.error("Email not found. Please try again. 电邮错误，请再尝试。")
                except Exception as e:
                    st.error(f"Error sending OTP: {e}")

        # Step 2: Verify OTP
        elif not st.session_state["verified"]:
            otp = st.text_input("Enter OTP 输入密码", type="password")
            if st.button("Verify OTP 验证密码"):
                if validate_otp(st.session_state["user_id"], otp):
                    st.session_state["verified"] = True
                    st.success("OTP verified successfully! Click 'Enter App'. 验证成功！请按 ‘进入’ 键。")
                    st.button("Enter App 进入", on_click=lambda: st.session_state.update({"logged_in": True}))
                else:
                    st.error("Invalid or expired OTP. Please request a new OTP.  密码错误/逾期,请再领取新密码。")
                    # Reset the OTP flow to allow retry
                    st.button("Resend new OTP 重新发送密码")
                    st.session_state["otp_sent"] = False
                    st.session_state["user_id"] = None

    else:
        st.sidebar.button("Logout 登出", on_click=logout_user)
        # Checkboxes for indicators
        st.write("选股条件（股票必须满足所有条件）：")
        st.write("Select indicators (stocks must meet all selected criteria):")
        col1, col2 = st.columns(2)

        with col1:
            rainbow_selected = st.checkbox("彩图", key="rainbow_check",
                                           value=st.session_state['criteria'].get('rainbow', False))
            n1_selected = st.checkbox("牛一", key="n1_check", value=st.session_state['criteria'].get('n1', False))
            y1_selected = st.checkbox("第一黄柱", key="y1_check", value=st.session_state['criteria'].get('y1', False))

        with col2:
            zj_selected = st.checkbox("资金所向", key="zj_check", value=st.session_state['criteria'].get('zj', False))
            qs_selected = st.checkbox("趋势专家", key="qs_check", value=st.session_state['criteria'].get('qs', False))

        # st.write("Filter by user-defined values (optional):")

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
            'rainbow': rainbow_selected,
            'n1': n1_selected,
            'y1': y1_selected,
            'zj': zj_selected,
            'qs': qs_selected,
            'min_volume': min_volume,
            'min_price': min_price,
            'min_banker_value': min_banker_value,
            'max_banker_value': max_banker_value,
        }

        col5, col6 = st.columns([1, 1])

        with col5:
            if st.button("选股 Screen Stock"):
                st.session_state['show_list'] = False
                try:
                    # Authenticate and download the file content (returns a DataFrame)
                    service = authenticate_drive_api()
                    file_id = '1dcLwOQ47kIW8NZJy0qkmQtknz6I4cTyO'
                    file_content = download_file(service, file_id)

                    if isinstance(file_content, pd.DataFrame) and not file_content.empty:
                        matching_symbols, temp_file_path = check_indicators_and_save(
                            file_content, min_volume, min_price, min_banker_value, max_banker_value)

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
                    with open(st.session_state['temp_file_path'], "r") as file:
                        file_data = file.read()
                    if file_data:
                        st.download_button(
                            label="下载名单 Download List",
                            data=file_data,
                            file_name=f"filtered_result_{today_date}.txt",
                            mime="text/plain"
                        )
                except FileNotFoundError:
                    st.warning("No file available to download.")

        # Manual stock symbol input
        stock_input = st.text_input("或输入股票代码以查看图表： Or enter a stock symbol for chart viewing:")
        if stock_input:
            select_stock(stock_input.strip())
            st.session_state['selected_stock'] = stock_input.strip()

        # Define a mapping for display labels
        criteria_labels = {
            "rainbow": "彩图",
            "n1": "牛一",
            "y1": "第一黄柱",
            "zj": "资金所向",
            "qs": "趋势专家",
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
