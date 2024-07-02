import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# Function to convert string values to bps
def convert_to_bps(value):
    if 'Mb/s' in value:
        return abs(float(value.replace(' Mb/s', '')) * 1e6)
    elif 'kb/s' in value:
        return abs(float(value.replace(' kb/s', '')) * 1e3)
    elif 'b/s' in value:
        return abs(float(value.replace(' b/s', '')))
    else:
        return 0.0

# Function to denormalize predictions
def denormalize(predictions, column_name, scaler, scaled_columns):
    predictions = predictions.flatten()
    min_val = scaler.data_min_[scaled_columns.index(column_name)]
    max_val = scaler.data_max_[scaled_columns.index(column_name)]
    denorm_predictions = predictions * (max_val - min_val) + min_val
    return denorm_predictions

# Streamlit app
st.title("Prediksi Kecepatan Internet")

# CSV File Uploader
st.subheader("CSV File Uploader:")
uploaded_file = st.file_uploader("Upload CSV file:", type=['csv'])

if uploaded_file is not None:
    delimiters = (',', ';')
    df = pd.read_csv(uploaded_file, sep='[,;]', engine='python')

    # Display the first few rows of the dataframe to ensure it is read correctly
    st.write(df.head())

    # Dropdowns for selecting time, input, and output columns
    st.subheader("Select Time, Input, and Output columns:")
    time_column = st.selectbox("Select Time column:", df.columns)
    input_column = st.selectbox("Select Input column:", df.columns)
    output_column = st.selectbox("Select Output column:", df.columns)

    # Convert input and output columns to bps
    df[input_column] = df[input_column].apply(convert_to_bps)
    df[output_column] = df[output_column].apply(convert_to_bps)

    # Display the first few rows to ensure the conversion is correct
    st.write(df.head())

    # Normalize the input column
    scaler = MinMaxScaler()
    df[[input_column]] = scaler.fit_transform(df[[input_column]])

    # Display the first few rows to ensure the normalization is correct
    st.write(df.head())

    # Split data into features and target
    X = df[[input_column]]
    y = df[output_column]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape features for LSTM input
    X_train = X_train.values.reshape((X_train.shape[0], 1, 1))
    X_test = X_test.values.reshape((X_test.shape[0], 1, 1))

    # Define LSTM model architecture
    def build_lstm_model():
        model = Sequential([
            LSTM(80, activation='relu', input_shape=(1, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(80, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    model = build_lstm_model()

    # Train and evaluate LSTM model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Predictions and evaluation
    predictions = model.predict(X_test)
    denorm_predictions = denormalize(predictions, output_column, scaler, [input_column])
    denorm_actual = denormalize(y_test.values, output_column, scaler, [input_column])
    mse = mean_squared_error(denorm_actual, denorm_predictions)

    # Display MSE
    st.write(f'MSE for {input_column} predicting {output_column}:', mse)

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 5))
    plt.plot(denorm_actual, label='Actual')
    plt.plot(denorm_predictions, label='Predicted')
    plt.title(f'Actual vs Predicted for {input_column} predicting {output_column}')
    plt.legend()
    st.pyplot(plt)

else:
    st.write("Silakan unggah file CSV.")
