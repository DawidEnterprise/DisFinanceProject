import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
import streamlit as st
import tensorflow as tf

# Set the Seaborn style to a dark grid background
sns.set(style="darkgrid")

# Streamlit app
st.title("Stock Price Analysis")

# User inputs
stock_symbol = st.text_input("Enter the stock symbol (e.g., AAPL):")
start_date = st.date_input("Select the start date:", pd.to_datetime('2023-03-01'))
end_date = st.date_input("Select the end date:", pd.to_datetime('2023-08-15'))

# Set the timeframe to 1 day
timeframe = '1d'

# Download the stock data with error handling
try:
    df = yf.download(stock_symbol, start=start_date, end=end_date, interval=timeframe)
except Exception as e:
    st.error(f"An error occurred while fetching the data: {str(e)}")
    df = None  # Set df to None to indicate no data

if df is not None and not df.empty:

    df['Pivot_Point'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Create a column to indicate the direction of the pivot point
    df['Pivot_Direction'] = df['Pivot_Point'].diff().apply(lambda x: 'Up' if x > 0 else 'Down')

    # Adjust sensitivity by changing the rolling window size for calculating SMA
    rolling_window_size = 20  # Adjust this value to decrease sensitivity
    df['SMA_20'] = df['Close'].rolling(window=rolling_window_size).mean()

    # Calculate the Relative Strength Index (RSI)
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=14).mean()
    average_loss = loss.rolling(window=14).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    df['RSI'] = rsi

    # Prepare data for LSTM model
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Function to prepare data for LSTM
    def create_dataset(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps, 0])
        return np.array(X), np.array(y)

    # Set time steps for LSTM
    time_steps = 60

    # Create the LSTM dataset
    X, y = create_dataset(scaled_data, time_steps)

    # Reshape data for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the LSTM model with improved settings
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dense(units=50))
    model.add(Dense(units=1))

    # Adjust the learning rate of the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Compile the model with the new optimizer
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model with more epochs
    model.fit(X, y, epochs=20, batch_size=32)

    # Make predictions for the future
    future_time_steps = 10  # Predicting the next 5 days
    future_predictions = []  # Store future predictions
    last_sequence = scaled_data[-time_steps:].reshape(1, -1)  # Last sequence in the data

    for i in range(future_time_steps):
        future_prediction = model.predict(last_sequence.reshape(1, time_steps, 1))
        future_predictions.append(future_prediction[0])
        last_sequence = np.append(last_sequence[:, 1:], future_prediction, axis=1)

    # Inverse transform the predicted prices
    future_predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Extend the index for predicted prices
    index_dates = pd.date_range(start=df.index[-1], periods=future_time_steps, freq='D')

    # Create a figure with two subplots using gridspec
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    # Plot the main chart (Close Price, 20-Day SMA, and Pivot Points)
    ax0 = plt.subplot(gs[0])
    ax0.plot(df.index, df['Close'], label='Close Price', linewidth=2)
    ax0.plot(df.index, df['SMA_20'], label='20-Day SMA', linestyle='--')

    # Add Buy/Sell labels based on crossing the 20-Day SMA
    buy_sell_labels = []
    for i in range(1, len(df)):
        if df['Close'][i] > df['SMA_20'][i] and df['Close'][i - 1] <= df['SMA_20'][i - 1]:
            buy_sell_labels.append('Buy')
        elif df['Close'][i] < df['SMA_20'][i] and df['Close'][i - 1] >= df['SMA_20'][i - 1]:
            buy_sell_labels.append('Sell')
        else:
            buy_sell_labels.append('')

    # Label 'Buy' or 'Sell' on the main chart
    for i, label in enumerate(buy_sell_labels):
        if label == 'Buy':
            ax0.annotate('Buy', xy=(df.index[i], df['Close'][i]), textcoords='offset points', xytext=(-5, -10), ha='center', color='green', fontsize=10, fontweight='bold')
        elif label == 'Sell':
            ax0.annotate('Sell', xy=(df.index[i], df['Close'][i]), textcoords='offset points', xytext=(-5, 10), ha='center', color='red', fontsize=10, fontweight='bold')

    # Plot the future predicted prices in red and label it
    ax0.plot(index_dates, future_predicted_prices, linestyle='-', color='red', label='Predicted Future Price')

    ax0.set_ylabel('Price')
    ax0.legend()
    ax0.grid(True)

    # Show up and down arrows for pivot points
    for index, row in df.iterrows():
        if row['Pivot_Direction'] == 'Up':
            ax0.annotate('^', xy=(index, row['Pivot_Point']), textcoords='offset points', xytext=(0, 10), ha='center')
        else:
            ax0.annotate('v', xy=(index, row['Pivot_Point']), textcoords='offset points', xytext=(0, -20), ha='center')

    ax0.set_title(f"{stock_symbol} Stock Price with Pivot Points and 20-Day SMA")
    ax0.set_ylabel('Price')
    ax0.legend()
    ax0.grid(True)

    # Remove x-axis labels from the main chart
    ax0.set_xticks([])
    ax0.set_xlabel('')

    # Plot the RSI in a separate subplot without sharing the x-axis
    ax1 = plt.subplot(gs[1])
    ax1.plot(df.index, df['RSI'], label='RSI', linestyle='-', color='red')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('RSI')
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()

    # Display the Matplotlib plot using Streamlit
    st.pyplot(fig)
elif df is not None:
    st.warning("No data available for the selected date range.")
