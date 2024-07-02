import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import streamlit as st

# Read data from CSV with UTF-8 encoding
file_path = "input2.csv"
try:
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8', errors='replace')

# Convert 'Time' column to datetime
df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M')

# Convert values to numeric, removing units like 'b/s', 'kb/s', 'Mb/s'
def convert_to_bps(value):
    if isinstance(value, str):
        if 'Mb/s' in value:
            return float(value.replace(' Mb/s', '')) * 1e6
        elif 'kb/s' in value:
            return float(value.replace(' kb/s', '')) * 1e3
        elif 'b/s' in value:
            return float(value.replace(' b/s', ''))
        else:
            return 0.0
    return value

# Convert all columns except 'Time'
for col in df.columns[1:]:
    df[col] = df[col].apply(convert_to_bps)

# Extract hour from 'Time' column
df['Hour'] = df['Time'].dt.hour

# Aggregate data for different floors
df['In_Lantai3'] = df['In - sfp3 - lantai 3']
df['Out_Lantai3'] = df['Out - sfp3 - lantai 3']
df['Total_Lantai3'] = df['In_Lantai3'] + df['Out_Lantai3']

df['In_Lantai7'] = df['In - sfp-sfpplus2 - lantai 7']
df['Out_Lantai7'] = df['Out - sfp-sfpplus2 - lantai 7']
df['Total_Lantai7'] = df['In_Lantai7'] + df['Out_Lantai7']

in_columns = [col for col in df.columns if col.startswith('In -') and 'lantai 3' not in col and 'lantai 7' not in col]
out_columns = [col for col in df.columns if col.startswith('Out -') and 'lantai 3' not in col and 'lantai 7' not in col]
df['In_Other'] = df[in_columns].sum(axis=1)
df['Out_Other'] = df[out_columns].sum(axis=1)
df['Total_Other'] = df['In_Other'] + df['Out_Other']

# Filter data for specific hours
target_hours = [8, 11, 14]
df_target = df[df['Hour'].isin(target_hours)]

# Normalize data
scaler = MinMaxScaler()
df_target[['In_Lantai3', 'Out_Lantai3', 'In_Lantai7', 'Out_Lantai7', 'In_Other', 'Out_Other']] = scaler.fit_transform(
    df_target[['In_Lantai3', 'Out_Lantai3', 'In_Lantai7', 'Out_Lantai7', 'In_Other', 'Out_Other']]
)

# Prepare data for LSTM
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps - 1):
        a = data[i:(i + time_steps), 0]
        X.append(a)
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Create datasets for each floor
time_steps = 3
X_lantai3, y_lantai3 = create_dataset(df_target[['In_Lantai3']].values, time_steps)
X_lantai7, y_lantai7 = create_dataset(df_target[['In_Lantai7']].values, time_steps)
X_other, y_other = create_dataset(df_target[['In_Other']].values, time_steps)

# Reshape input to be [samples, time steps, features]
X_lantai3 = np.reshape(X_lantai3, (X_lantai3.shape[0], X_lantai3.shape[1], 1))
X_lantai7 = np.reshape(X_lantai7, (X_lantai7.shape[0], X_lantai7.shape[1], 1))
X_other = np.reshape(X_other, (X_other.shape[0], X_other.shape[1], 1))

# Split data into train and test sets
X_train_lantai3, X_test_lantai3, y_train_lantai3, y_test_lantai3 = train_test_split(X_lantai3, y_lantai3, test_size=0.2, random_state=42)
X_train_lantai7, X_test_lantai7, y_train_lantai7, y_test_lantai7 = train_test_split(X_lantai7, y_lantai7, test_size=0.2, random_state=42)
X_train_other, X_test_other, y_train_other, y_test_other = train_test_split(X_other, y_other, test_size=0.2, random_state=42)

# Build LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and evaluate model
def train_evaluate_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=1):
    model = build_model()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = sqrt(mse)
    return model, history, predictions, rmse

# Train models
model_lantai3, history_lantai3, predictions_lantai3, rmse_lantai3 = train_evaluate_model(X_train_lantai3, y_train_lantai3, X_test_lantai3, y_test_lantai3)
model_lantai7, history_lantai7, predictions_lantai7, rmse_lantai7 = train_evaluate_model(X_train_lantai7, y_train_lantai7, X_test_lantai7, y_test_lantai7)
model_other, history_other, predictions_other, rmse_other = train_evaluate_model(X_train_other, y_train_other, X_test_other, y_test_other)

# Streamlit app
st.title("Prediksi Kecepatan Internet dengan LSTM")

# Display RMSE
st.write(f"RMSE Lantai 3: {rmse_lantai3:.2f}")
st.write(f"RMSE Lantai 7: {rmse_lantai7:.2f}")
st.write(f"RMSE Lantai Lainnya: {rmse_other:.2f}")

# Plot loss
def plot_loss(history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)

plot_loss(history_lantai3, "Loss for Lantai 3")
plot_loss(history_lantai7, "Loss for Lantai 7")
plot_loss(history_other, "Loss for Lantai Lainnya")

# Predict for specific times
times = [8, 11, 14]
times_encoded = np.array(times).reshape(-1, 1)

predictions_in_lantai3 = model_lantai3.predict(times_encoded)
predictions_out_lantai3 = model_lantai3.predict(times_encoded)

predictions_in_lantai7 = model_lantai7.predict(times_encoded)
predictions_out_lantai7 = model_lantai7.predict(times_encoded)

predictions_in_other = model_other.predict(times_encoded)
predictions_out_other = model_other.predict(times_encoded)

# Display predictions
def display_predictions(title, predictions_in, predictions_out):
    st.subheader(title)
    for time, pred_in, pred_out in zip(times, predictions_in, predictions_out):
        st.write(f'{time}:00:')
        st.write(f'  In: {pred_in[0] / 1e6:.2f} Mbps')
        st.write(f'  Out: {pred_out[0] / 1e6:.2f} Mbps')
        st.write(f'  Total: {(pred_in[0] + pred_out[0]) / 1e6:.2f} Mbps')

display_predictions("Prediksi Untuk Lantai 3", predictions_in_lantai3, predictions_out_lantai3)
display_predictions("Prediksi Untuk Lantai 7", predictions_in_lantai7, predictions_out_lantai7)
display_predictions("Prediksi Untuk Semua Lantai", predictions_in_other, predictions_out_other)

