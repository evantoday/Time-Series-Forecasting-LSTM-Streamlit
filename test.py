import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


#convert function
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
    df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M')

    # Mengonversi kolom 'Time' menjadi datetime
    # df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M')
    with st.container(border=True):
        st.subheader("Feature Selection:")
        st.divider()
        date_f = st.selectbox(label="Select Date Feature:",options=df.columns)
            
        input_f = st.selectbox(label="Input Features (X):",options=[element for element in list(df.columns) if element != date_f])

        output_f = st.selectbox(label="Output Feature (Y):",options=[element for element in list(df.columns) if element != date_f])

    # Menampilkan beberapa baris pertama untuk memastikan data terbaca dengan benar
    st.write(df.head())

    # Mengubah nilai menjadi numerik, menghilangkan satuan seperti 'b/s', 'kb/s', 'Mb/s'
    for col in df.columns[1:]:
        df[col] = df[col].apply(convert_to_bps)

    # Menampilkan beberapa baris pertama untuk memastikan perubahan
    st.write(df.head())

    # Mengambil jam dari kolom 'Time'
    df['Hour'] = df['Time'].dt.hour

    # Mengambil data untuk lantai 3
    df['In_Lantai3'] = df['In - sfp3 - lantai 3']
    df['Out_Lantai3'] = df['Out - sfp3 - lantai 3']
    df['Total_Lantai3'] = df['In_Lantai3'] + df['Out_Lantai3']

    # Mengambil data untuk lantai 7
    df['In_Lantai7'] = df['In - sfp-sfpplus2 - lantai 7']
    df['Out_Lantai7'] = df['Out - sfp-sfpplus2 - lantai 7']
    df['Total_Lantai7'] = df['In_Lantai7'] + df['Out_Lantai7']

    # Mengambil data untuk semua lantai kecuali lantai 3 dan 7
    in_columns = [col for col in df.columns if col.startswith('In -') and 'lantai 3' not in col and 'lantai 7' not in col]
    out_columns = [col for col in df.columns if col.startswith('Out -') and 'lantai 3' not in col and 'lantai 7' not in col]
    df['In_Other'] = df[in_columns].sum(axis=1)
    df['Out_Other'] = df[out_columns].sum(axis=1)
    df['Total_Other'] = df['In_Other'] + df['Out_Other']

    # Mengambil data pada jam tertentu
    target_hours = [8, 11, 14]
    df_target = df[df['Hour'].isin(target_hours)]

    # Mengkonversi jam menjadi fitur numerik
    label_encoder = LabelEncoder()
    df_target['Hour_Encoded'] = label_encoder.fit_transform(df_target['Hour'])

    # Normalize the data
    scaled_columns = ['In_Lantai3', 'Out_Lantai3', 'In_Lantai7', 'Out_Lantai7', 'In_Other', 'Out_Other']
    scaler = MinMaxScaler()
    df_target[scaled_columns] = scaler.fit_transform(df_target[scaled_columns])

    # Split data into features and target for Lantai 3
    X_lantai3 = df_target[['Hour_Encoded']]
    y_in_lantai3 = df_target['In_Lantai3']
    y_out_lantai3 = df_target['Out_Lantai3']

    # Split into training and test sets for Lantai 3
    X_train_in_lantai3, X_test_in_lantai3, y_train_in_lantai3, y_test_in_lantai3 = train_test_split(X_lantai3, y_in_lantai3, test_size=0.2, random_state=42)
    X_train_out_lantai3, X_test_out_lantai3, y_train_out_lantai3, y_test_out_lantai3 = train_test_split(X_lantai3, y_out_lantai3, test_size=0.2, random_state=42)

    # Split data into features and target for Lantai 7
    y_in_lantai7 = df_target['In_Lantai7']
    y_out_lantai7 = df_target['Out_Lantai7']

    # Split into training and test sets for Lantai 7
    X_train_in_lantai7, X_test_in_lantai7, y_train_in_lantai7, y_test_in_lantai7 = train_test_split(X_lantai3, y_in_lantai7, test_size=0.2, random_state=42)
    X_train_out_lantai7, X_test_out_lantai7, y_train_out_lantai7, y_test_out_lantai7 = train_test_split(X_lantai3, y_out_lantai7, test_size=0.2, random_state=42)

    # Split data into features and target for Other Lantai
    y_in_other = df_target['In_Other']
    y_out_other = df_target['Out_Other']

    # Split into training and test sets for Other Lantai
    X_train_in_other, X_test_in_other, y_train_in_other, y_test_in_other = train_test_split(X_lantai3, y_in_other, test_size=0.2, random_state=42)
    X_train_out_other, X_test_out_other, y_train_out_other, y_test_out_other = train_test_split(X_lantai3, y_out_other, test_size=0.2, random_state=42)

    # Define LSTM model architecture for In and Out separately
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

    model_in_lantai3 = build_lstm_model()
    model_out_lantai3 = build_lstm_model()
    model_in_lantai7 = build_lstm_model()
    model_out_lantai7 = build_lstm_model()
    model_in_other = build_lstm_model()
    model_out_other = build_lstm_model()

    # Reshape features for LSTM input
    X_train_in_lantai3 = X_train_in_lantai3.values.reshape((X_train_in_lantai3.shape[0], 1, 1))
    X_test_in_lantai3 = X_test_in_lantai3.values.reshape((X_test_in_lantai3.shape[0], 1, 1))
    X_train_out_lantai3 = X_train_out_lantai3.values.reshape((X_train_out_lantai3.shape[0], 1, 1))
    X_test_out_lantai3 = X_test_out_lantai3.values.reshape((X_test_out_lantai3.shape[0], 1, 1))
    X_train_in_lantai7 = X_train_in_lantai7.values.reshape((X_train_in_lantai7.shape[0], 1, 1))
    X_test_in_lantai7 = X_test_in_lantai7.values.reshape((X_test_in_lantai7.shape[0], 1, 1))
    X_train_out_lantai7 = X_train_out_lantai7.values.reshape((X_train_out_lantai7.shape[0], 1, 1))
    X_test_out_lantai7 = X_test_out_lantai7.values.reshape((X_test_out_lantai7.shape[0], 1, 1))
    X_train_in_other = X_train_in_other.values.reshape((X_train_in_other.shape[0], 1, 1))
    X_test_in_other = X_test_in_other.values.reshape((X_test_in_other.shape[0], 1, 1))
    X_train_out_other = X_train_out_other.values.reshape((X_train_out_other.shape[0], 1, 1))
    X_test_out_other = X_test_out_other.values.reshape((X_test_out_other.shape[0], 1, 1))

    # Train and evaluate LSTM models
    def train_evaluate_model(model, X_train, y_train, X_test, y_test, title):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])

        predictions = model.predict(X_test)
        denorm_predictions = denormalize(predictions, y_test.name, scaler, scaled_columns)
        denorm_actual = denormalize(y_test.values, y_test.name, scaler, scaled_columns)
        mse = mean_squared_error(denorm_actual, denorm_predictions)
        
        st.write(f'MSE {title}:', mse)
        
        # Plot predictions vs actual values
        plt.figure(figsize=(10, 5))
        plt.plot(denorm_actual, label='Actual')
        plt.plot(denorm_predictions, label='Predicted')
        plt.title(f'Actual vs Predicted {title}')
        plt.legend()
        st.pyplot(plt)

    train_evaluate_model(model_in_lantai3, X_train_in_lantai3, y_train_in_lantai3, X_test_in_lantai3, y_test_in_lantai3, 'In Lantai 3')
    train_evaluate_model(model_out_lantai3, X_train_out_lantai3, y_train_out_lantai3, X_test_out_lantai3, y_test_out_lantai3, 'Out Lantai 3')
    train_evaluate_model(model_in_lantai7, X_train_in_lantai7, y_train_in_lantai7, X_test_in_lantai7, y_test_in_lantai7, 'In Lantai 7')
    train_evaluate_model(model_out_lantai7, X_train_out_lantai7, y_train_out_lantai7, X_test_out_lantai7, y_test_out_lantai7, 'Out Lantai 7')
    train_evaluate_model(model_in_other, X_train_in_other, y_train_in_other, X_test_in_other, y_test_in_other, 'In Other')
    train_evaluate_model(model_out_other, X_train_out_other, y_train_out_other, X_test_out_other, y_test_out_other, 'Out Other')
else:
    st.write("Silakan unggah file CSV.")


