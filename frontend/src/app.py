import streamlit as st
import requests
import pandas as pd
import pytz
from datetime import datetime, timedelta
import plotly.graph_objs as go

# Timezones
ny_tz = pytz.timezone("America/New_York")
lj_tz = pytz.timezone("Europe/Ljubljana")

def fetch_data(url, method="GET", json_body=None):
    if method == "GET":
        resp = requests.get(url)
    else:
        resp = requests.post(url, json=json_body)
    resp.raise_for_status()
    return resp.json()

def convert_time(dt_str):
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    dt_ny = ny_tz.localize(dt)
    dt_lj = dt_ny.astimezone(lj_tz)
    return dt_lj

def create_dataframe(data):
    rows = data["data"]["values"]
    df = pd.DataFrame(rows)
    df["datetime"] = df["datetime"].apply(convert_time)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    return df

def show_user_page():
    st.title("AAPL Stock Prices: Real + Predicted (Multiple points)")

    stock_json = fetch_data("http://localhost:8000/stock/apple")
    predict_json = fetch_data("http://localhost:8000/predict/price", method="POST", json_body=stock_json)

    df_real = create_dataframe(stock_json)
    last_dt = df_real.index[-1]

    predictions = predict_json["prediction"]
    num_preds = len(predictions)

    pred_times = [last_dt + timedelta(minutes=15 * (i + 1)) for i in range(num_preds)]
    df_pred = pd.DataFrame(predictions, index=pred_times)
    for col in ["open", "high", "low", "close"]:
        df_pred[col] = pd.to_numeric(df_pred[col])

    metrics = ["open", "high", "low", "close"]

    for metric in metrics:
        st.subheader(f"{metric.capitalize()} Price")

        real_trace = go.Scatter(
            x=df_real.index,
            y=df_real[metric],
            mode='lines+markers',
            name='Real',
            line=dict(color='blue')
        )

        pred_trace = go.Scatter(
            x=df_pred.index,
            y=df_pred[metric],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', dash='dash'),
            marker=dict(symbol='diamond', size=8)
        )

        fig = go.Figure(data=[real_trace, pred_trace])
        fig.update_layout(
            xaxis_title='Datetime (Ljubljana)',
            yaxis_title=f'{metric.capitalize()} Price',
            legend=dict(x=0, y=1),
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Trading activity block at bottom with spinner
    st.subheader("Trading Activity")
    with st.spinner("Loading trading activity..."):
        try:
            activity = fetch_data("http://localhost:8000/predict/activity", method="POST", json_body=stock_json)
            label = activity.get("label", "N/A")
            st.success(f"**Activity Level:** {label}")
        except Exception as e:
            st.error(f"Error fetching trading activity: {e}")

def show_admin_page():
    st.title("Admin dashboard")

    try:
        result = fetch_data("http://localhost:8000/metrics/price/calculate")
        print("Response from /metrics/price/calculate:", result)
    except requests.exceptions.HTTPError as http_err:
        response = http_err.response
        status_code = response.status_code if response else "No response"
        content = response.text if response else "No content"
        print(f"HTTP error: {status_code} - {content}")
    except Exception as e:
        print(f"Error calling /metrics/price/calculate: {e}")

    metrics_json = fetch_data("http://localhost:8000/metrics/price")

    st.markdown("### Model Performance Metrics for AAPL Price Prediction")
    st.write(f"**Symbol:** {metrics_json.get('symbol', 'N/A')}")
    st.write(f"**Model Type:** {metrics_json.get('model_type', 'N/A')}")

    st.write("---")
    st.write(f"**Mean Absolute Error (MAE):** {metrics_json.get('mae', 'N/A'):.4f}")
    st.write(f"**Mean Squared Error (MSE):** {metrics_json.get('mse', 'N/A'):.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {metrics_json.get('rmse', 'N/A'):.4f}")

def show_reports_page():
    st.title("Data Validation & Testing Reports")

    st.markdown("### Data Validation Report")
    st.markdown(
        """
        <iframe src="https://stock-predictor-data-validation.netlify.app"
                width="100%" height="600" frameborder="0"></iframe>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Data Testing Report")
    st.markdown(
        """
        <iframe src="https://stock-predictor-data-testing.netlify.app/data_testing_report.html"
                width="100%" height="600" frameborder="0"></iframe>
        """,
        unsafe_allow_html=True
    )

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["User View", "Admin Metrics", "Reports"])

    if page == "User View":
        show_user_page()
    elif page == "Admin Metrics":
        show_admin_page()
    elif page == "Reports":
        show_reports_page()

if __name__ == "__main__":
    main()
