# 📈 AI-Powered Stock Price Prediction System

A sophisticated machine learning application that uses LSTM neural networks to predict stock prices with real-time analysis and comprehensive visualizations.

## 🚀 Features

- **Real-time Stock Data**: Fetches live stock data from Yahoo Finance
- **LSTM Neural Network**: Advanced deep learning model for time series prediction
- **Technical Indicators**: RSI, Moving Averages, Volatility analysis
- **Interactive Charts**: Price trends, volume analysis, and prediction visualizations
- **Performance Metrics**: RMSE, MAE scores for model accuracy
- **Responsive Design**: Professional UI with mobile-friendly interface
- **Deterministic Predictions**: Consistent results with reproducible AI models

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning model
- **NumPy & Pandas** - Data processing
- **scikit-learn** - Data preprocessing
- **yfinance** - Stock data API
- **Matplotlib** - Chart generation

### Frontend
- **HTML5 & CSS3**
- **JavaScript** 
- **Bootstrap** - Responsive design
- **Professional UI Components**

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for stock data

### Step 1: Clone the Repository
```bash
git clone https://github.com/Dadipavan/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create requirements.txt
```txt
Flask==2.3.3
tensorflow==2.13.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
yfinance==0.2.21
Werkzeug==2.3.7
```

## 🚀 Usage

### Running the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Prediction System

1. **Navigate to Homepage**: Open your browser and go to `http://localhost:5000`

2. **Enter Stock Symbol**: Type a valid stock ticker (e.g., AAPL, GOOGL, TSLA)

3. **Get Prediction**: Click "Predict Stock Price" to generate analysis

4. **View Results**: 
   - Next day price prediction
   - Current market metrics
   - Technical indicators
   - Performance charts
   - Historical price analysis

## 📊 Features Breakdown

### 1. Stock Data Analysis
- **Current Price**: Real-time stock price
- **Predicted Price**: AI-generated next-day prediction
- **Price Change**: Dollar and percentage change
- **52-Week High/Low**: Annual price range

### 2. Technical Indicators
- **RSI (Relative Strength Index)**: Momentum oscillator (0-100)
- **Moving Averages**: 20-day and 50-day trends
- **Volatility**: Price movement analysis
- **Trading Volume**: Market activity metrics

### 3. AI Model Performance
- **Training RMSE**: Root Mean Square Error on training data
- **Test RMSE**: Model accuracy on unseen data
- **Training MAE**: Mean Absolute Error metrics
- **Test MAE**: Prediction accuracy scores

### 4. Interactive Charts
- **Price Chart**: Stock price with moving averages
- **RSI Chart**: Overbought/oversold indicators
- **Volume Chart**: Trading activity visualization
- **Prediction Chart**: Actual vs predicted comparison

### 5. Price History
- **Last 10 Days**: Recent price movements
- **Daily Changes**: Percentage change analysis
- **Trend Analysis**: Price direction indicators

## 🔧 Configuration

### Model Parameters
```python
# LSTM Architecture
LOOKBACK_DAYS = 60        # Historical days for prediction
LSTM_UNITS = 50           # Neural network complexity
EPOCHS = 50               # Training iterations
BATCH_SIZE = 32           # Training batch size
TRAIN_SPLIT = 0.8         # 80% training, 20% testing
```

### Environment Variables
```python
# TensorFlow Configuration
TF_ENABLE_ONEDNN_OPTS = '0'
TF_CPP_MIN_LOG_LEVEL = '2'
TF_DETERMINISTIC_OPS = '1'
PYTHONHASHSEED = '42'
```

## 📈 Model Architecture

### LSTM Neural Network
```
Input Layer (60 timesteps, 1 feature)
    ↓
LSTM Layer 1 (50 units, return_sequences=True)
    ↓
LSTM Layer 2 (50 units)
    ↓
Dense Output Layer (1 unit)
    ↓
Linear Activation (Price Prediction)
```

### Data Pipeline
```
Raw Stock Data → Technical Indicators → Data Scaling → 
Sequence Creation → LSTM Training → Prediction → 
Performance Analysis → Visualization
```

## 📋 API Endpoints

### Home Page
```
GET /
Returns: Main application interface
```

### Stock Prediction
```
POST /predict
Parameters: ticker (stock symbol)
Returns: Prediction results with charts and analysis
```

## 🔍 Supported Stock Symbols

The application supports all major stock exchanges:
- **NYSE**: New York Stock Exchange
- **NASDAQ**: Technology stocks
- **International**: Major global stocks

### Example Tickers
- **Technology**: AAPL, GOOGL, MSFT, TSLA, NVDA
- **Finance**: JPM, BAC, WFC, GS
- **Healthcare**: JNJ, PFE, UNH, ABBV
- **Energy**: XOM, CVX, COP
- **ETFs**: SPY, QQQ, IWM, GLD

## 🚨 Error Handling

### Common Issues
1. **Invalid Ticker**: App validates stock symbol format
2. **No Data Available**: Handles delisted or invalid stocks
3. **Insufficient Data**: Requires minimum 100 data points
4. **Network Issues**: Graceful handling of API failures
5. **Model Errors**: Fallback mechanisms for training issues

### Error Messages
- Clear, user-friendly error descriptions
- Helpful suggestions for resolution
- Maintains application stability

## 🔒 Security Features

- **Input Validation**: Prevents malicious ticker inputs
- **Data Sanitization**: Clean data processing
- **Error Isolation**: Prevents crashes from bad data
- **Resource Management**: Memory cleanup and optimization

## 📊 Performance Optimization

### Speed Improvements
- **Efficient Data Processing**: Optimized pandas operations
- **Memory Management**: Automatic cleanup of matplotlib figures
- **Batch Processing**: Optimized neural network training
- **Caching Strategy**: Deterministic results without redundant processing

### Accuracy Features
- **Deterministic Training**: Reproducible results
- **Data Quality Checks**: Validates input data integrity
- **Model Validation**: Train/test split for accuracy assessment
- **Technical Analysis**: Multiple indicators for comprehensive analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include error handling for new features
- Test with multiple stock symbols
- Update documentation for new features

## 📝 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free stock data API
- **TensorFlow Team** for the excellent deep learning framework
- **Flask Community** for the lightweight web framework
- **Matplotlib** for powerful visualization capabilities
- **scikit-learn** for machine learning utilities

## 📞 Support

### Getting Help
- **Issues**: Report bugs via GitHub Issues
- **Documentation**: Check this README for common questions
- **Community**: Join discussions in GitHub Discussions

### Contact Information
- **Email**: dadisaipavan1514@gmail.com
- **GitHub**: [@Dadipavan](https://github.com/Dadipavan)
- **LinkedIn**: [Valavala Sai Pavan](https://www.linkedin.com/in/valavala-dadi-naga-siva-sai-pavan-b6a1a728b/)

## 🔮 Future Enhancements

### Planned Features
- [ ] **Database Integration** (PostgreSQL)
- [ ] **User Authentication** (Login system)


## 📈 Version History

### v1.0.0 (Current)
- Initial release with LSTM prediction
- Technical indicators integration
- Interactive chart generation
- Responsive web interface
- Deterministic model training
- Comprehensive error handling

---

**⭐ If you found this project helpful, please give it a star on GitHub!**

**💡 Have suggestions or found a bug? Please open an issue or submit a pull request.**


**🚀 Ready to predict the future of stocks? Let's get started!**

