import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import streamlit as st
import plotly.graph_objs as go

def load_data(ticker):
    try:
        data = yf.download(ticker, period='max', progress=False)
        if data.empty:
            st.error("The data was not loaded correctly. Please check the ticker and try again.")
            return None
        data = data.reset_index()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def predict_future_prices_polynomial(data, days_ahead, degree=2):
    if data is None or data.empty:
        return None
    
    data['Day'] = np.arange(len(data))
    
    X = data[['Day']]
    y = data['Close']
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_scaled)
    
    model = LinearRegression()
    model.fit(X_poly, y_scaled)
    
    future_days = np.arange(len(data), len(data) + days_ahead).reshape(-1, 1)
    future_days_scaled = scaler_X.transform(future_days)
    future_days_poly = poly.transform(future_days_scaled)
    
    future_prices_scaled = model.predict(future_days_poly)
    future_prices = scaler_y.inverse_transform(future_prices_scaled)
    
    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days_ahead)
    future_data = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_prices.flatten()})
    return future_data

def plot_trend_with_prediction_polynomial(data, period, degree=2):
    days_selected_map = {'5d': 5, '1mo': 30, '3mo': 90, '1y': 365, '2y': 731, '5y': 1827}
    days_selected = days_selected_map[period]
    
    data_filtered = data.iloc[-days_selected:]

    future_data = predict_future_prices_polynomial(data, days_selected, degree)
    
    if future_data is None:
        return
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data_filtered['Date'], y=data_filtered['Close'],
        mode='lines', name="Cryptocurrency Price Analysis",
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=future_data['Date'], y=future_data['Predicted Close'],
        mode='lines', name="Choose the prediction period:",
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title=f"Prevision for next {days_selected} Dias",
        xaxis_title='Data',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0, y=1, traceorder='normal')
    )

    st.plotly_chart(fig)

# Streamlit app
st.title('Cryptography - Streamlit app')
ticker = st.text_input("Enter the cryptocurrency ticker (e.g., BTC-USD):", "BTC-USD")
period = st.selectbox("Choice period:", ['5d', '1mo', '3mo', '1y', '2y', '5y'])

if st.button("Load Data"):
    data = load_data(ticker)
    if data is not None:
        st.write("Load data sucess!")
        plot_trend_with_prediction_polynomial(data, period, degree=2)