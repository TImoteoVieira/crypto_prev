import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import streamlit as st
import plotly.graph_objs as go

def load_data(ticker):
    try:
        # Desativando a barra de progresso ao fazer o download dos dados
        data = yf.download(ticker, period='max', progress=False)
        if data.empty:
            st.error("Os dados não foram carregados corretamente. Verifique o ticker e tente novamente.")
            return None
        data = data.reset_index()  # Resetando o índice para criar a coluna 'Date'
        return data
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")
        return None

def predict_future_prices_polynomial(data, days_ahead, degree=2):
    if data is None or data.empty:
        return None
    
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
    days_selected_map = {'5d': 5, '1mo': 30, '3mo': 90, '1y': 365, '2y': 731, '5y': 1827}
    days_selected = days_selected_map[period]
    
    # Filtrar os últimos 'days_selected' dias e garantir que a coluna 'Date' esteja presente
    data_filtered = data.iloc[-days_selected:]

    future_data = predict_future_prices_polynomial(data, days_selected, degree)
    
    if future_data is None:
        return
    
    # Criar um gráfico interativo com Plotly
    fig = go.Figure()

    # Adicionar linha do preço histórico
    fig.add_trace(go.Scatter(
        x=data_filtered['Date'], y=data_filtered['Close'],  # Usando a coluna 'Date' criada
        mode='lines', name='Preço Histórico',
        line=dict(color='blue')
    ))

    # Adicionar linha da previsão futura
    fig.add_trace(go.Scatter(
        x=future_data['Date'], y=future_data['Predicted Close'],
        mode='lines', name='Tendência Futura',
        line=dict(color='orange', dash='dash')
    ))

    # Configurar layout
    fig.update_layout(
        title=f"Previsão de Preço para os Próximos {days_selected} Dias (Polinomial de Grau {degree})",
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0, y=1, traceorder='normal')
    )

    # Exibir gráfico no Streamlit
    st.plotly_chart(fig)

# Streamlit app
st.title('Análise de Preço de Criptomoedas')
ticker = st.text_input("Digite o ticker da criptomoeda (ex: BTC-USD):", "BTC-USD")
period = st.selectbox("Escolha o período de previsão:", ['5d', '1mo', '3mo', '1y', '2y', '5y'])

if st.button("Carregar Dados"):
    data = load_data(ticker)
    if data is not None:
        st.write("Dados carregados com sucesso!")
        plot_trend_with_prediction_polynomial(data, period, degree=2)