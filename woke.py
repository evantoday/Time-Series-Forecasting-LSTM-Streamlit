# -*- coding: utf-8 -*-
"""Bismillah ini yang terbaik.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xuQJKAbboHevyqxUD-04jQ6_DYObtENI
"""
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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

# Function to build LSTM model
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

# Function to plot predictions
def plot_predictions(ax, title, actual_in, actual_out, predictions_in, predictions_out, times):
    ax.plot(times, [actual / 1e6 for actual in actual_in], marker='o', label='Actual In', color='blue')
    ax.plot(times, [actual / 1e6 for actual in actual_out], marker='o', label='Actual Out', color='cyan')
    ax.plot(times, [pred_in / 1e6 for pred_in in predictions_in], marker='o', label='Predicted In', color='red')
    ax.plot(times, [pred_out / 1e6 for pred_out in predictions_out], marker='o', label='Predicted Out', color='orange')
    ax.plot(times, [(actual_in[i] + actual_out[i]) / 1e6 for i in range(len(actual_in))], marker='o', label='Actual Total', color='green')
    ax.plot(times, [(pred_in + pred_out) / 1e6 for pred_in, pred_out in zip(predictions_in, predictions_out)], marker='o', label='Predicted Total', color='purple')
    ax.set_title(title)
    ax.set_xlabel('Jam')
    ax.set_ylabel('Kecepatan Internet (Mbps)')
    ax.legend()

# Streamlit app
st.title("Prediksi Kecepatan Internet")

# CSV File Uploader
st.subheader("CSV File Uploader:")
uploaded_file = st.file_uploader("Upload CSV file:", type=['csv'])
if uploaded_file is not None:
    delimiters = (',', ';')
    df = pd.read_csv(uploaded_file, sep='[,;]', engine='python')
    df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M')

    # Display initial data
    st.write(df.head())

    # Convert values to bps
    for col in df.columns[1:]:
        df[col] = df[col].apply(convert_to_bps)

    # Display converted data
    st.write(df.head())

    # Extract hour from Time column
    df['Hour'] = df['Time'].dt.hour

    # List of user input features
    user_input_features = [
        'In - ether1', 'In - sfp-sfpplus4 - modem', 'In - l2tp-out1', 'In - sfp1 - lantai 1', 'In - sfp2 - lantai 2', 
        'In - sfp3 - lantai 3', 'In - sfp4 - lantai 4', 'In - sfp5 - lantai 5', 'In - sfp-sfpplus1 - lantai 6', 
        'In - sfp-sfpplus2 - lantai 7', 'In - sfp-sfpplus3', 'Out - ether1', 'Out - sfp-sfpplus4 - modem', 
        'Out - l2tp-out1', 'Out - sfp1 - lantai 1', 'Out - sfp2 - lantai 2', 'Out - sfp3 - lantai 3', 
        'Out - sfp4 - lantai 4', 'Out - sfp5 - lantai 5', 'Out - sfp-sfpplus1 - lantai 6', 'Out - sfp-sfpplus2 - lantai 7', 
        'Out - sfp-sfpplus3'
    ]

    # Select input and output features from user input list
    input_f = st.selectbox("Select Input Feature (X):", user_input_features)
    output_f = st.selectbox("Select Output Feature (Y):", user_input_features)
    df['In_UserInput'] = df[input_f]
    df['Out_UserInput'] = df[output_f]
    df['Total_UserInput'] = df['In_UserInput'] + df['Out_UserInput']

    in_columns = [col for col in df.columns if col.startswith('In -') and input_f not in col]
    out_columns = [col for col in df.columns if col.startswith('Out -') and output_f not in col]
    df['In_Other'] = df[in_columns].sum(axis=1)
    df['Out_Other'] = df[out_columns].sum(axis=1)
    df['Total_Other'] = df['In_Other'] + df['Out_Other']

    # Select target hours based on user input
    target_hours = st.multiselect('Select Target Hours:', df['Hour'].unique().tolist())
    
    if st.button('Start Training') and target_hours:
        df_target = df[df['Hour'].isin(target_hours)]

        # Encode hour feature
        label_encoder = LabelEncoder()
        df_target['Hour_Encoded'] = label_encoder.fit_transform(df_target['Hour'])

        # Normalize the data
        scaled_columns = ['In_UserInput', 'Out_UserInput', 'In_Other', 'Out_Other']
        scaler = MinMaxScaler()
        df_target[scaled_columns] = scaler.fit_transform(df_target[scaled_columns])

        # Split data into features and targets
        X = df_target[['Hour_Encoded']]
        y_features = {
            'UserInput': ('In_UserInput', 'Out_UserInput'),
        }

        models = {}
        predictions = {}
        denorm_predictions = {}
        rmse = {}

        for key, (y_in, y_out) in y_features.items():
            X_train_in, X_test_in, y_train_in, y_test_in = train_test_split(X, df_target[y_in], test_size=0.2, random_state=42)
            X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X, df_target[y_out], test_size=0.2, random_state=42)

            # Reshape features for LSTM input
            X_train_in = X_train_in.values.reshape((X_train_in.shape[0], 1, 1))
            X_test_in = X_test_in.values.reshape((X_test_in.shape[0], 1, 1))
            X_train_out = X_train_out.values.reshape((X_train_out.shape[0], 1, 1))
            X_test_out = X_test_out.values.reshape((X_test_out.shape[0], 1, 1))

            # Build and train models
            models[key] = {
                'in': build_lstm_model(),
                'out': build_lstm_model()
            }

            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

            models[key]['in'].fit(X_train_in, y_train_in, epochs=100, batch_size=1, verbose=0, callbacks=[early_stopping])
            models[key]['out'].fit(X_train_out, y_train_out, epochs=100, batch_size=1, verbose=0, callbacks=[early_stopping])

            # Make predictions
            pred_in = models[key]['in'].predict(X_test_in)
            pred_out = models[key]['out'].predict(X_test_out)

            # Denormalize predictions
            denorm_pred_in = denormalize(pred_in, y_in, scaler, scaled_columns)
            denorm_pred_out = denormalize(pred_out, y_out, scaler, scaled_columns)

            predictions[key] = {
                'in': pred_in,
                'out': pred_out
            }

            denorm_predictions[key] = {
                'in': denorm_pred_in,
                'out': denorm_pred_out
            }

            # Denormalize actual values for plotting
            denorm_actual_in = denormalize(y_test_in.values, y_in, scaler, scaled_columns)
            denorm_actual_out = denormalize(y_test_out.values, y_out, scaler, scaled_columns)

            # Calculate RMSE
            rmse[key] = {
                'in': mean_squared_error(denorm_actual_in, denorm_pred_in, squared=False),
                'out': mean_squared_error(denorm_actual_out, denorm_pred_out, squared=False)
            }

        # Display denormalized predictions
        st.subheader("Hasil Denormalisasi Prediksi")
        times = target_hours
        times_str = [f'{time}:00' for time in times]

        for key in y_features.keys():
            st.write(f"Denormalized predictions for {key}:")
            for time, pred_in, pred_out in zip(times_str, denorm_predictions[key]['in'][:len(times)], denorm_predictions[key]['out'][:len(times)]):
                st.write(f'Predicted internet speed at {time}:')
                st.write(f'  In: {pred_in / 1e6:.2f} Mbps')
                st.write(f'  Out: {pred_out / 1e6:.2f} Mbps')

        # Plot predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, (y_in, y_out) in y_features.items():
            denorm_actual_in = denormalize(y_test_in.values, y_in, scaler, scaled_columns)
            denorm_actual_out = denormalize(y_test_out.values, y_out, scaler, scaled_columns)
            plot_predictions(ax, f'Prediksi of user input', denorm_actual_in[:len(times)], denorm_actual_out[:len(times)], denorm_predictions[key]['in'][:len(times)], denorm_predictions[key]['out'][:len(times)], times_str)

        plt.tight_layout()
        st.pyplot(fig)

        # Print all RMSE values
        st.write("RMSE Values:")
        for key in y_features.keys():
            st.write(f"{key} In: {rmse[key]['in']}")
            st.write(f"{key} Out: {rmse[key]['out']}")

        # Calculate and print total RMSE
        total_rmse = np.sqrt(np.mean([rmse[key]['in']**2 + rmse[key]['out']**2 for key in y_features.keys()]))
        st.write("Total RMSE:", total_rmse)
