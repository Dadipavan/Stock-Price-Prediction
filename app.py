import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import warnings
import tensorflow as tf
import random
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Set ALL random seeds for complete reproducibility
RANDOM_SEED = 42

def set_global_seeds():
    """Set all possible random seeds for reproducibility"""
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Configure TensorFlow for deterministic behavior
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass  # Fallback for older TensorFlow versions

# Set seeds immediately
set_global_seeds()

app = Flask(__name__)

def fetch_stock_data(ticker, period='2y'):
    """Fetch stock data fresh every time"""
    try:
        print(f"Fetching fresh data for {ticker}")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data available for {ticker}")
        
        # Add technical indicators
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # Remove any NaN values
        df = df.dropna()
        
        return df
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {str(e)}")

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_lstm_data(data, lookback=60):
    """Prepare data for LSTM with deterministic scaling"""
    # Reset seeds for consistency
    set_global_seeds()
    
    # Ensure data is clean
    data = np.array(data).astype(np.float32)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def create_lstm_model():
    """Create optimized LSTM model with maximum determinism"""
    # Reset seeds before model creation
    set_global_seeds()
    
    # Create model with fixed architecture
    inputs = Input(shape=(60, 1))
    
    # First LSTM layer with deterministic initializers
    x = LSTM(50, 
             return_sequences=True,
             activation='tanh',
             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED),
             recurrent_initializer=tf.keras.initializers.Orthogonal(seed=RANDOM_SEED),
             bias_initializer=tf.keras.initializers.Zeros())(inputs)
    
    # Second LSTM layer
    x = LSTM(50,
             activation='tanh',
             kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED),
             recurrent_initializer=tf.keras.initializers.Orthogonal(seed=RANDOM_SEED),
             bias_initializer=tf.keras.initializers.Zeros())(x)
    
    # Output layer
    outputs = Dense(1,
                   activation='linear',
                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=RANDOM_SEED),
                   bias_initializer=tf.keras.initializers.Zeros())(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use deterministic optimizer with fixed parameters
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def train_and_predict(ticker, df):
    """Train model and make prediction - fresh every time"""
    print(f"Training fresh model for {ticker}")
    
    # Reset all seeds for consistency
    set_global_seeds()
    
    # Prepare data for LSTM
    data = df['Close'].values
    
    if len(data) < 100:
        raise ValueError(f"Insufficient data for {ticker}. Need at least 100 data points.")
    
    X, y, scaler = prepare_lstm_data(data)
    
    if len(X) == 0:
        raise ValueError(f"No training data available for {ticker}")
    
    # Split data for training and validation (80-20 split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train model
    model = create_lstm_model()
    
    # Train with deterministic settings
    set_global_seeds()
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0,
        shuffle=False  # Important for reproducibility
    )
    
    # Make predictions for validation
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate accuracy metrics
    train_rmse = float(np.sqrt(mean_squared_error(y_train_actual, train_pred)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test_actual, test_pred)))
    train_mae = float(mean_absolute_error(y_train_actual, train_pred))
    test_mae = float(mean_absolute_error(y_test_actual, test_pred))
    
    # Prepare all actual and predicted prices for visualization
    all_actual = np.concatenate([y_train_actual.flatten(), y_test_actual.flatten()])
    all_predicted = np.concatenate([train_pred.flatten(), test_pred.flatten()])
    
    # Make prediction for next day
    last_60_days = data[-60:].reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60_days)
    X_pred = last_60_scaled.reshape(1, 60, 1)
    
    # Reset seed before final prediction for consistency
    set_global_seeds()
    next_price_scaled = model.predict(X_pred, verbose=0)
    next_price = float(scaler.inverse_transform(next_price_scaled)[0, 0])
    
    return {
        'next_price': next_price,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'all_actual': all_actual,
        'all_predicted': all_predicted,
        'model': model,
        'scaler': scaler
    }

def create_comprehensive_charts(df, ticker, actual_prices=None, predicted_prices=None):
    """Create multiple charts for comprehensive analysis"""
    charts = {}
    
    # Set consistent style
    plt.style.use('default')
    set_global_seeds()  # For consistent colors
    
    try:
        # 1. Price Chart with Moving Averages
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot last 100 days
        recent_data = df.tail(100)
        
        ax.plot(recent_data.index, recent_data['Close'], 
                label='Close Price', linewidth=2.5, color='#1f77b4')
        
        if not recent_data['MA_20'].isna().all():
            ax.plot(recent_data.index, recent_data['MA_20'], 
                    label='20-day MA', linewidth=1.8, color='#ff7f0e', alpha=0.8)
        
        if not recent_data['MA_50'].isna().all():
            ax.plot(recent_data.index, recent_data['MA_50'], 
                    label='50-day MA', linewidth=1.8, color='#2ca02c', alpha=0.8)
        
        ax.set_title(f'{ticker} Stock Price Analysis (Last 100 Days)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        charts['price'] = save_plot_to_base64()
        
        # 2. RSI Chart
        fig, ax = plt.subplots(figsize=(14, 5))
        
        recent_rsi = recent_data['RSI'].dropna()
        if len(recent_rsi) > 0:
            ax.plot(recent_rsi.index, recent_rsi.values, 
                    color='purple', linewidth=2.5)
            ax.axhline(y=70, color='r', linestyle='--', alpha=0.8, 
                      linewidth=1.5, label='Overbought (70)')
            ax.axhline(y=30, color='g', linestyle='--', alpha=0.8, 
                      linewidth=1.5, label='Oversold (30)')
            ax.fill_between(recent_rsi.index, 30, 70, alpha=0.1, color='gray')
            
            ax.set_title('Relative Strength Index (RSI)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel('RSI', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        charts['rsi'] = save_plot_to_base64()
        
        # 3. Volume Chart
        if 'Volume' in df.columns and not df['Volume'].isna().all():
            fig, ax = plt.subplots(figsize=(14, 5))
            
            # Last 30 days volume
            volume_data = df.tail(30)
            
            # Color bars based on price movement
            colors = []
            for i in range(len(volume_data)):
                if volume_data['Close'].iloc[i] >= volume_data['Open'].iloc[i]:
                    colors.append('green')
                else:
                    colors.append('red')
            
            ax.bar(volume_data.index, volume_data['Volume'], 
                  color=colors, alpha=0.7, width=0.8)
            ax.set_title('Trading Volume (Last 30 Days)', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel('Volume', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Format volume numbers
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            charts['volume'] = save_plot_to_base64()
        
        # 4. Prediction vs Actual Chart
        if actual_prices is not None and predicted_prices is not None:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Show last 50 points for clarity
            show_points = min(50, len(actual_prices))
            
            x_range = range(show_points)
            ax.plot(x_range, actual_prices[-show_points:], 
                   label='Actual Prices', linewidth=2.5, color='blue', alpha=0.8)
            ax.plot(x_range, predicted_prices[-show_points:], 
                   label='Predicted Prices', linewidth=2.5, color='red', alpha=0.7)
            
            ax.set_title('Model Performance: Actual vs Predicted Prices', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            charts['prediction'] = save_plot_to_base64()
    
    except Exception as e:
        print(f"Error creating charts: {e}")
        # Create empty chart on error
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Chart generation error: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        charts['error'] = save_plot_to_base64()
    
    return charts

def save_plot_to_base64():
    """Save plot to base64 string"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close('all')  # Close all figures to free memory
    return img_str

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = None
    try:
        ticker = request.form.get('ticker', '').upper().strip()
        
        if not ticker:
            raise ValueError("Please enter a valid ticker symbol")
        
        # Validate ticker format
        if not ticker.isalpha() or len(ticker) > 10:
            raise ValueError("Invalid ticker symbol format")
        
        print(f"Processing prediction for {ticker}")
        
        # Fetch fresh data every time
        df = fetch_stock_data(ticker)
        
        # Train model and get prediction - fresh every time
        prediction_result = train_and_predict(ticker, df)
        
        next_price = prediction_result['next_price']
        
        # Calculate statistics
        current_price = float(df['Close'].iloc[-1])
        price_change = next_price - current_price
        price_change_percent = (price_change / current_price) * 100
        
        # Create comprehensive charts
        charts = create_comprehensive_charts(
            df, ticker,
            prediction_result['all_actual'],
            prediction_result['all_predicted']
        )
        
        # Prepare statistics with proper error handling
        stats = {
            'current_price': current_price,
            'predicted_price': next_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'train_rmse': prediction_result['train_rmse'],
            'test_rmse': prediction_result['test_rmse'],
            'train_mae': prediction_result['train_mae'],
            'test_mae': prediction_result['test_mae'],
            'rsi': float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0,
            'ma_20': float(df['MA_20'].iloc[-1]) if not pd.isna(df['MA_20'].iloc[-1]) else current_price,
            'ma_50': float(df['MA_50'].iloc[-1]) if not pd.isna(df['MA_50'].iloc[-1]) else current_price,
            'volatility': float(df['Volatility'].iloc[-1]) if not pd.isna(df['Volatility'].iloc[-1]) else 0.0,
            'volume': int(df['Volume'].iloc[-1]) if 'Volume' in df.columns and not pd.isna(df['Volume'].iloc[-1]) else 0,
            'high_52w': float(df['High'].rolling(252).max().iloc[-1]) if len(df) >= 252 else float(df['High'].max()),
            'low_52w': float(df['Low'].rolling(252).min().iloc[-1]) if len(df) >= 252 else float(df['Low'].min())
        }
        
        # FIXED: Last 10 days data with proper error handling
        last_10_days = []
        try:
            df_last = df.tail(11)  # Get 11 days to calculate changes for last 10
            
            for i in range(1, min(11, len(df_last))):
                current_idx = -len(df_last) + i
                previous_idx = current_idx - 1
                
                current_price_day = float(df.iloc[current_idx]['Close'])
                previous_price_day = float(df.iloc[previous_idx]['Close'])
                
                change_percent = ((current_price_day - previous_price_day) / previous_price_day) * 100
                
                last_10_days.append({
                    'date': df.index[current_idx].strftime('%d-%m-%y'),
                    'price': current_price_day,
                    'change': change_percent
                })
            
            # Reverse to show most recent first
            last_10_days = last_10_days[::-1]
            
        except Exception as e:
            print(f"Error processing last 10 days data: {e}")
            # Fallback: simple last 10 days without changes
            last_10_days = []
            for i in range(-10, 0):
                if abs(i) <= len(df):
                    last_10_days.append({
                        'date': df.index[i].strftime('%d-%m-%y'),
                        'price': float(df['Close'].iloc[i]),
                        'change': 0.0
                    })
        
        print(f"Prediction completed for {ticker}: ${next_price:.2f}")
        
        return render_template('predict.html',
                             ticker=ticker,
                             stats=stats,
                             charts=charts,
                             last_10_days=last_10_days,
                             success=True)
                             
    except Exception as e:
        error_msg = f"Error processing {ticker if ticker else 'request'}: {str(e)}"
        print(f"Error: {error_msg}")
        return render_template('predict.html',
                             ticker=ticker,
                             error=error_msg,
                             success=False)

if __name__ == '__main__':
    app.run(debug=True)