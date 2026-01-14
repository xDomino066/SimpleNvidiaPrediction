
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import time

# --- KONFIGURACJA ---
TICKER = 'NVDA'
SEQ_LENGTH = 60  # Ile dni wstecz widzi model
TRAIN_SPLIT_DATE = '2025-01-01'
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# --- FIX NA ŚCIEŻKĘ ZAPISU ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'nvda_lstm_best.pth')
print(f"\n📂 Model will be saved to: {MODEL_PATH}")

# Sprawdzenie GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🚀 System check:")
print(f"   Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# --- 1. POBIERANIE DANYCH ---
def get_data(ticker):
    print(f"\n📥 Downloading FULL history for {ticker} (since 1999)...")
    df = yf.download(ticker, start='1999-01-22', progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Wskaźniki
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # Target: Gap otwarcia (Open_next - Close_today) / Close_today
    df['Target'] = (df['Open'].shift(-1) - df['Close']) / df['Close']

    df.dropna(inplace=True)
    return df


# --- 2. PRZYGOTOWANIE DANYCH ---
def create_sequences(features, targets, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features[i:(i + seq_length)]
        y = targets[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# --- 3. MODEL LSTM ---
class NVDA_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(NVDA_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# --- START ---
if __name__ == "__main__":
    # Dane
    df = get_data(TICKER)

    feature_cols = ['Close', 'SMA_50', 'SMA_200', 'RSI']
    target_col = 'Target'

    # Podział
    train_df = df[df.index < TRAIN_SPLIT_DATE].copy()
    test_df = df[df.index >= TRAIN_SPLIT_DATE].copy()

    # Normalizacja
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train_raw = scaler_X.fit_transform(train_df[feature_cols])
    y_train_raw = scaler_y.fit_transform(train_df[[target_col]])

    X_test_raw = scaler_X.transform(test_df[feature_cols])
    y_test_raw = scaler_y.transform(test_df[[target_col]])

    # Sekwencje
    X_train, y_train = create_sequences(X_train_raw, y_train_raw, SEQ_LENGTH)
    X_test, y_test = create_sequences(X_test_raw, y_test_raw, SEQ_LENGTH)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Inicjalizacja
    model = NVDA_LSTM(input_dim=len(feature_cols), hidden_dim=128, output_dim=1, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- TRENING ---
    print("\n🔥 Starting Training GPU...")
    print("-" * 65)
    print(f"{'Epoch':<10} | {'Avg Loss':<15} | {'RMSE':<10} | {'Status':<15}")
    print("-" * 65)

    best_loss = float('inf')

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        rmse_score = np.sqrt(avg_loss)

        save_msg = ""
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            save_msg = "💾 New Best!"

        print(f"Ep {epoch + 1}/{EPOCHS:<3} | {avg_loss:.6f}        | {rmse_score:.6f}   | {save_msg}")

    print("-" * 65)
    print(f"🏆 Best Loss: {best_loss:.6f}")

    # --- EWALUACJA I KONWERSJA NA DOLARY (USD) ---
    print("\n🔄 Generating price chart in USD...")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        test_inputs = torch.from_numpy(X_test).float().to(device)
        predictions_scaled = model(test_inputs).cpu().numpy()

        # 1. Odwracamy skalowanie, żeby dostać procenty
        predictions_pct = scaler_y.inverse_transform(predictions_scaled).flatten()

        # 2. Pobieramy prawdziwe ceny z danych testowych
        # Uwaga: Przez SEQ_LENGTH pierwsze 60 dni testowych "zjada" tworzenie sekwencji
        test_data_slice = test_df.iloc[SEQ_LENGTH:]

        real_close_prices = test_data_slice['Close'].values
        real_open_prices = test_data_slice['Open'].values
        dates = test_data_slice.index

        # 3. Obliczamy PRZEWIDZIANĄ CENĘ w USD
        # Wzór: Predicted_Open = Wczorajsze_Close * (1 + Model_Predicted_Percent)
        predicted_prices_usd = real_close_prices * (1 + predictions_pct)

    # --- WYKRES ---
    plt.figure(figsize=(14, 7))

    # Rzeczywista cena
    plt.plot(dates, real_open_prices, label='Real Price (USD)', color='black', alpha=0.6, linewidth=2)

    # Przewidziana cena przez AI
    plt.plot(dates, predicted_prices_usd, label='AI Predicted Price (USD)', color='#00ff00', linewidth=1.5, alpha=0.9)

    plt.title(f'NVIDIA Price Prediction: Real vs AI (Test Data: 2025+)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()