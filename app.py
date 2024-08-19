import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import streamlit as st
import matplotlib.dates as mdates

def load_data(ticker):
    data = yf.download(ticker, period='max')
    if data.empty:
        st.error("Os dados não foram carregados corretamente. Verifique o ticker e tente novamente.")
        return None
    return data

def predict_future_prices_polynomial(data, days_ahead, degree=2):
    if data is None or data.empty:
        return None
    
    data = data.reset_index()
    data['Day'] = np.arange(len(data))
    
    X = data[['Day']]
    y = data['Close']
    
    # Normalizar os dados
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    # Regressão polinomial
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_scaled)
    
    model = LinearRegression()
    model.fit(X_poly, y_scaled)
    
    # Previsão dos próximos dias
    future_days = np.arange(len(data), len(data) + days_ahead).reshape(-1, 1)
    future_days_scaled = scaler_X.transform(future_days)
    future_days_poly = poly.transform(future_days_scaled)
    
    future_prices_scaled = model.predict(future_days_poly)
    future_prices = scaler_y.inverse_transform(future_prices_scaled)
    
    future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days_ahead)
    future_data = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_prices.flatten()})
    return future_data

def plot_trend_with_prediction_polynomial(data, period, degree=2):
    days_selected_map = {'5d': 5, '1 mês': 30, '3 meses': 90, '1 ano': 365, '2 anos': 731, '5 anos': 1827}
    days_selected = days_selected_map[period]
    
    data_filtered = data.iloc[-days_selected:]
    
    future_data = predict_future_prices_polynomial(data, days_selected, degree)
    
    if future_data is None:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plotar os dados históricos filtrados
    plt.plot(data_filtered.index, data_filtered['Close'], label="Preço Histórico", color='blue', linewidth=2)
    
    # Plotar os dados de previsão
    plt.plot(future_data['Date'], future_data['Predicted Close'], label="Tendência Futura", color='orange', linestyle='--', linewidth=2)
    
    plt.fill_between(future_data['Date'], future_data['Predicted Close'], color='orange', alpha=0.1)
    
    # Ajustar a formatação das datas no eixo X de acordo com o período selecionado
    if period == '5d':
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif period == '1mo' or period == '3mo':
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif period == '1y':
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.gcf().autofmt_xdate(rotation=45)
    
    plt.xlabel("Data")
    plt.ylabel("Preço (USD)")
    plt.title(f"Previsão de Preço para os Próximos {days_selected} Dias (Polinomial de Grau {degree})")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Streamlit app
st.title('Análise de Preço de Criptomoedas')
ticker = st.text_input("Digite o ticker da criptomoeda (ex: BTC-USD):", "BTC-USD")
period = st.selectbox("Escolha o período de previsão:", ['5 dias', '1 mês', '3 meses', '1 ano', '2 anos', '5 anos'])

if st.button("Carregar Dados"):
    data = load_data(ticker)
    if data is not None:
        st.write("Dados carregados com sucesso!")
        plot_trend_with_prediction_polynomial(data, period, degree=2)  # Começar com grau 2 e ajustar conforme necessário
