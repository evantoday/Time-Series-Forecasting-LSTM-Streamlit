# Import necessary libraries
import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from torch import nn
from sklearn.metrics import mean_squared_error
import time
from datetime import timedelta
import gc
import plotly.figure_factory as ff

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Multi-Step Time Series Forecasting LSTM", page_icon="https://github.com/harshitv804/Time-Series-Forecasting-LSTM-Streamlit/assets/100853494/1f137778-ef9c-45e1-87dd-a3ca6bd79b46")

# Set the device to CUDA if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define columns for layout
col3, col4 = st.columns([1, 5])
col1, col2 = st.columns([3, 2])

# Define the LSTM model class
class LSTMForecasting(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, linear_hidden_size, lstm_num_layers, linear_num_layers, output_size):
        super(LSTMForecasting, self).__init__()
        self.linear_hidden_size = linear_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.linear_num_layers = linear_num_layers
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.linear_layers = nn.ModuleList()
        self.linear_num_layers -= 1
        self.linear_layers.append(nn.Linear(self.lstm_hidden_size, self.linear_hidden_size))

        for _ in range(linear_num_layers):
            self.linear_layers.append(nn.Linear(self.linear_hidden_size, int(self.linear_hidden_size / 1.5)))
            self.linear_hidden_size = int(self.linear_hidden_size / 1.5)

        self.fc = nn.Linear(self.linear_hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        for linear_layer in self.linear_layers:
            out = linear_layer(out)

        out = self.fc(out[:, -1, :])
        return out

# Initialize session state variables if not already initialized
if 'sd_click' not in st.session_state:
    st.session_state.sd_click = False

if 'train_click' not in st.session_state:
    st.session_state.train_click = False

if 'disable_opt' not in st.session_state:
    st.session_state.disable_opt = False

if 'model_save' not in st.session_state:
    st.session_state.model_save = None

# Function to split the data into input and output sequences
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).float()
# Button click handlers
def onClickSD():
    st.session_state.sd_click = True

def onClickTrain():
    st.session_state.train_click = True
def preprocess_data(df):
    # Convert all columns to numeric, forcing errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    # Interpolate missing values
    df = df.interpolate(method='linear').bfill().ffill()
    return df

# Example data loading (replace this with your actual data loading method)
# df = pd.read_csv('your_file.csv')

# Assuming 'training_df' is your DataFrame
# Apply preprocessing
training_df = preprocess_data(training_df)

# Ensure 'training_df.values' is entirely numeric
print(training_df.dtypes)

# Define your n_steps_in and n_steps_out
n_steps_in = 10  # Example value
n_steps_out = 5  # Example value

# Call split_sequences function
X, y = split_sequences(training_df.values, n_steps_in, n_steps_out)
# Function to preprocess the data
def preProcessData(date_f, input_f, output_f):
    preProcessDataList = input_f
    preProcessDataList.insert(-1, output_f)
    preProcessDF = df[list(dict.fromkeys(preProcessDataList))]
    
    # Convert all columns to numeric, forcing errors to NaN
    preProcessDF = preProcessDF.apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN values with a forward fill method, then back fill if necessary
    preProcessDF = preProcessDF.interpolate(method='linear').bfill().ffill()

    # Ensure date column is in datetime format
    preProcessDF.insert(0, date_f, df[date_f])
    if str(preProcessDF.at[0, date_f]).isdigit():
        preProcessDF[date_f] = pd.to_datetime(preProcessDF[date_f], format='%Y')
    else:
        preProcessDF[date_f] = pd.to_datetime(preProcessDF[date_f])
    
    return preProcessDF
# Function to check the frequency of the date feature
def check_date_frequency(date_series):
    dates = pd.to_datetime(date_series)

    differences = (dates - dates.shift(1)).dropna()

    daily_count = (differences == timedelta(days=1)).sum()
    hourly_count = (differences == timedelta(hours=1)).sum()
    weekly_count = (differences == timedelta(weeks=1)).sum()
    monthly_count = (differences >= timedelta(days=28, hours=23, minutes=59)).sum()  # Approximate 28 days to a month

    if daily_count > max(monthly_count, hourly_count, weekly_count):
        return 365
    elif monthly_count > max(daily_count, hourly_count, weekly_count):
        return 12
    elif weekly_count > max(daily_count, hourly_count, monthly_count):
        return 52
    elif hourly_count > max(daily_count, weekly_count, monthly_count):
        return 24*365  # Assuming hourly data is daily data repeated every hour
    else:
        return 1

# Function to convert network speed values to bits per second
def convert_to_bps(value):
    if 'Mb/s' in value:
        return abs(float(value.replace(' Mb/s', '')) * 1e6)
    elif 'kb/s' in value:
        return abs(float(value.replace(' kb/s', '')) * 1e3)
    elif 'b/s' in value:
        return abs(float(value.replace(' b/s', '')))
    else:
        return 0.0

# Function to perform seasonal decomposition and display results
def sea_decomp(date_f, input_f, output_f):
    if date_f:
        sea_decomp_data = preProcessData(date_f, input_f, output_f)
        corr_df = sea_decomp_data.select_dtypes(include=['int', 'float'])
        correlation_matrix = np.round(corr_df.corr(), 1)
        result = seasonal_decompose(sea_decomp_data.set_index(date_f)[output_f], model='additive', period=check_date_frequency(sea_decomp_data[date_f]))

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=result.seasonal.index.values, y=result.seasonal.values, mode='lines', line=dict(color='orange')))
        fig_s.update_layout(title='Seasonal',
                            xaxis_title='Date',
                            yaxis_title='Value',
                            height=300)

        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=result.trend.index.values, y=result.trend.values, mode='lines', line=dict(color='orange')))
        fig_t.update_layout(title='Trend',
                            xaxis_title='Date',
                            yaxis_title='Value',
                            height=300)

        fig_corr = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            colorscale='Viridis')

        with st.container(border=True):
            st.subheader("Correlation Matrix:")
            st.divider()
            st.plotly_chart(fig_corr, use_container_width=True)

        with st.container(border=True):
            st.subheader("Seasonal Decompose:")
            st.divider()
            st.plotly_chart(fig_t, use_container_width=True)
            st.divider()
            st.plotly_chart(fig_s, use_container_width=True)

        with st.container(border=True):
            st.subheader("Pre-Processed Data Preview:")
            st.divider()
            st.metric(label="Total Rows:", value=sea_decomp_data.shape[0])
            st.metric(label="Total Columns:", value=sea_decomp_data.shape[1])
            st.dataframe(sea_decomp_data, use_container_width=True, height=250)
        return sea_decomp_data

# Display the logo and title
with col3:
    st.image("https://github.com/harshitv804/Time-Series-Forecasting-LSTM-Streamlit/assets/100853494/1f137778-ef9c-45e1-87dd-a3ca6bd79b46")

with col4:
    st.title("Multi-Step Time Series Forecasting LSTM")
    st.subheader("Simple Streamlit GUI for LSTM Forecasting")

# File uploader and feature selection UI
with col1:
    with st.container(border=True):
        st.subheader("CSV File Uploader:")
        st.divider()
        uploaded_file = st.file_uploader("Upload CSV file:", type=['csv'])

    if uploaded_file is not None:
        delimiters = (',', ';')
        df = pd.read_csv(uploaded_file, sep='[,;]', engine='python')
        df['Time'] = pd.to_datetime(df['Time'], format='%d/%m/%Y %H:%M')
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(convert_to_bps)
        with st.container(border=True):
            st.subheader("Dataset Preview:")
            st.divider()
            st.dataframe(df, use_container_width=True, height=250)

        with st.container(border=True):
            st.subheader("Feature Selection:")
            st.divider()
            date_f = st.selectbox('Select Date Feature:', df.columns.tolist(), key=1)
            input_f = st.multiselect('Select Input Features:', df.columns.tolist(), key=2)
            output_f = st.selectbox('Select Output Feature:', df.columns.tolist(), key=3)

# Seasonal decomposition button
with col2:
    with st.container(border=True):
        st.subheader("Seasonal Decomposition:")
        st.divider()
        sea_decomp_btn = st.button("Run Seasonal Decomposition", on_click=onClickSD)

    if sea_decomp_btn or st.session_state.sd_click:
        sea_decomp(date_f, input_f, output_f)

# LSTM model training UI
with col1:
    with st.container(border=True):
        st.subheader("Train LSTM Model:")
        st.divider()
        if st.session_state.disable_opt:
            opt_epoch = 1
        else:
            opt_epoch = st.slider("Epochs:", min_value=1, max_value=1000, value=10, step=10)
        opt_lr = st.slider("Learning Rate:", min_value=0.0001, max_value=0.1, value=0.001, step=0.001, format="%.4f")
        opt_hidden_size = st.slider("Hidden Size:", min_value=1, max_value=128, value=64)
        opt_num_layer = st.slider("Number of Layers:", min_value=1, max_value=5, value=3)
        input_seq_len = st.slider("Input Sequence Length:", min_value=1, max_value=20, value=10)
        pred_seq_len = st.slider("Prediction Sequence Length:", min_value=1, max_value=20, value=5)
        opt_batch = st.slider("Batch Size:", min_value=1, max_value=64, value=32)

        with st.container():
            disable_optimizer_check = st.checkbox("Disable Optimizer", help="Disable optimizer if training with large data.")
            train_btn = st.button("Train Model", on_click=onClickTrain)

        if train_btn or st.session_state.train_click:
            with st.spinner("Training the model..."):
                st.session_state.train_click = True
                training_df = preProcessData(date_f, input_f, output_f)
                corr_df = training_df.select_dtypes(include=['int', 'float'])
                correlation_matrix = np.round(corr_df.corr(), 1)
                feature_num = len(input_f)
                out_feature_num = 1
                n_steps_in = input_seq_len
                n_steps_out = pred_seq_len

                X, y = split_sequences(training_df.values, n_steps_in, n_steps_out)

                scaler = MinMaxScaler()
                X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
                X = scaler.fit_transform(X)
                X = X.reshape((X.shape[0], n_steps_in, feature_num))
                y = scaler.fit_transform(y)

                dataset = torch.utils.data.TensorDataset(X, y)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt_batch, shuffle=False)

                model = LSTMForecasting(input_size=feature_num, lstm_hidden_size=opt_hidden_size, linear_hidden_size=opt_hidden_size, lstm_num_layers=opt_num_layer, linear_num_layers=opt_num_layer, output_size=out_feature_num)
                model = model.to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr)
                epoch_list = []
                loss_list = []

                for epoch in range(opt_epoch):
                    for i, (inputs, labels) in enumerate(dataloader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        epoch_list.append(epoch)
                        loss_list.append(loss.item())

                model.eval()
                y_true = []
                y_pred = []

                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(dataloader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        y_true += labels.cpu().numpy().tolist()
                        y_pred += outputs.cpu().numpy().tolist()

                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                mse = mean_squared_error(y_true, y_pred)
                st.success(f'Training Completed! MSE: {mse}')

                st.session_state.model_save = model

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=epoch_list, mode='lines', name='Epochs'))
                fig.add_trace(go.Scatter(y=loss_list, mode='lines', name='Loss'))

                with st.container(border=True):
                    st.subheader("Training Progress:")
                    st.divider()
                    st.plotly_chart(fig, use_container_width=True)

# Footer with link to the author
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
        }
    </style>
    <div class="footer">
        <p>Developed by <a href="https://github.com/harshitv804">Harshit Vardhan</a></p>
    </div>
""", unsafe_allow_html=True)
