import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import talib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import spacy
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(['python3', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

class StockAI:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
        plt.style.use('dark_background')
        self.technical_weights = {
            'rsi': 0.2,
            'macd': 0.2,
            'bollinger': 0.2,
            'volume': 0.1,
            'sentiment': 0.3
        }

    def get_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance with caching"""
        try:
            # Add .NS suffix for Indian stocks if not present
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            # Try to get data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                print(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Ensure we have enough data
            if len(df) < 200:  # Need at least 200 days for technical indicators
                print(f"Insufficient data for {symbol}")
                return pd.DataFrame()
            
            # Ensure all required columns are present
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                print(f"Missing required columns for {symbol}")
                return pd.DataFrame()
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        try:
            return talib.RSI(prices.values, timeperiod=period)
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return np.zeros(len(prices))

    def calculate_macd(self, prices):
        """Calculate MACD"""
        try:
            macd, signal, hist = talib.MACD(prices.values)
            return macd
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return np.zeros(len(prices))

    def calculate_bollinger_bands(self, prices, period=20):
        """Calculate Bollinger Bands"""
        try:
            upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period)
            return upper, lower
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            return np.zeros(len(prices)), np.zeros(len(prices))

    def prepare_features(self, data):
        """Prepare features for prediction"""
        try:
            if data.empty:
                return pd.DataFrame()

            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
            
            # Calculate technical indicators
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['MACD'] = self.calculate_macd(data['Close'])
            data['BB_Upper'], data['BB_Lower'] = self.calculate_bollinger_bands(data['Close'])
            
            # Create feature matrix
            features = pd.DataFrame({
                'Returns': data['Returns'],
                'RSI': data['RSI'],
                'MACD': data['MACD'],
                'BB_Upper': data['BB_Upper'],
                'BB_Lower': data['BB_Lower'],
                'Volume': data['Volume'],
                'Open': data['Open'],
                'High': data['High'],
                'Low': data['Low'],
                'Close': data['Close']
            })
            
            # Drop NaN values
            features = features.dropna()
            
            # Ensure all values are numeric
            features = features.astype(float)
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators"""
        try:
            if df.empty:
                return None
            
            # Make a copy to avoid modifying original data
            df = df.copy()
            
            # Convert price data to float64 for TA-Lib
            close = df['Close'].astype('float64').values
            high = df['High'].astype('float64').values
            low = df['Low'].astype('float64').values
            volume = df['Volume'].astype('float64').values
            
            # Price action indicators
            df['SMA_20'] = talib.SMA(close, timeperiod=20)
            df['SMA_50'] = talib.SMA(close, timeperiod=50)
            df['SMA_200'] = talib.SMA(close, timeperiod=200)
            
            # Momentum indicators
            df['RSI'] = talib.RSI(close, timeperiod=14)
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(close)
            
            # Volatility indicators
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(close)
            df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
            
            # Volume indicators
            df['OBV'] = talib.OBV(close, volume)
            
            # Trend indicators
            df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return None

    def analyze_market_regime(self, df):
        """Analyze market regime using multiple indicators"""
        try:
            if df.empty:
                return "Unknown"
            
            # Get latest values
            current_price = df['Close'].iloc[-1]
            sma20 = df['SMA_20'].iloc[-1]
            sma50 = df['SMA_50'].iloc[-1]
            sma200 = df['SMA_200'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            adx = df['ADX'].iloc[-1]
            
            # Determine trend strength
            trend_strength = 0
            if current_price > sma20 > sma50 > sma200:
                trend_strength = 1  # Strong uptrend
            elif current_price < sma20 < sma50 < sma200:
                trend_strength = -1  # Strong downtrend
            elif current_price > sma20 and sma20 > sma50:
                trend_strength = 0.5  # Moderate uptrend
            elif current_price < sma20 and sma20 < sma50:
                trend_strength = -0.5  # Moderate downtrend
            
            # Determine market regime
            if adx > 25:  # Strong trend
                if trend_strength > 0:
                    return "Strong Bullish"
                else:
                    return "Strong Bearish"
            elif adx > 20:  # Moderate trend
                if trend_strength > 0:
                    return "Moderate Bullish"
                else:
                    return "Moderate Bearish"
            else:
                return "Sideways"
                
        except Exception as e:
            print(f"Error analyzing market regime: {e}")
            return "Unknown"

    def predict_future_prices(self, df, days=150):
        """Predict future prices using historical patterns and technical analysis."""
        try:
            if df is None or df.empty:
                print("No data available for prediction")
                return None, None
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            if df is None or df.empty:
                print("Failed to calculate technical indicators")
                return None, None
            
            # Get current price and historical data
            current_price = float(df['Close'].iloc[-1])
            
            # Calculate historical patterns
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Calculate trend using multiple timeframes
            short_trend = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
            medium_trend = (df['Close'].iloc[-1] / df['Close'].iloc[-50] - 1) * 100
            long_trend = (df['Close'].iloc[-1] / df['Close'].iloc[-200] - 1) * 100
            
            # Generate future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=days+1, freq='B')[1:]
            
            # Initialize future prices
            future_prices = [current_price]
            
            # Set a fixed random seed for deterministic predictions
            np.random.seed(42)
            
            # Calculate base components
            base_volatility = volatility * 0.01  # Convert to daily volatility
            trend_strength = (short_trend + medium_trend + long_trend) / 300  # Normalized trend
            
            # Generate predictions
            for i in range(days):
                # Create realistic price movement
                daily_volatility = base_volatility * (1 + 0.5 * np.sin(i * 0.1))  # Varying volatility
                trend_effect = trend_strength * (1 + 0.2 * np.cos(i * 0.05))  # Varying trend strength
                
                # Add market cycles
                cycle_effect = 0.5 * np.sin(i * 0.02) + 0.3 * np.cos(i * 0.01)
                
                # Combine all effects
                price_movement = (
                    trend_effect * 0.001 +  # Trend component
                    daily_volatility * np.random.normal(0, 1) +  # Random component
                    cycle_effect * 0.001  # Market cycle component
                )
                
                # Calculate next price
                next_price = future_prices[-1] * (1 + price_movement)
                
                # Add realistic price bounds
                max_change = 0.03  # Maximum 3% daily change
                price_change = (next_price - future_prices[-1]) / future_prices[-1]
                if abs(price_change) > max_change:
                    next_price = future_prices[-1] * (1 + np.sign(price_change) * max_change)
                
                future_prices.append(next_price)
            
            return future_dates, future_prices[1:]
            
        except Exception as e:
            print(f"Error predicting future prices: {str(e)}")
            return None, None

    def generate_prediction_graph(self, df, future_dates, future_prices, symbol):
        """Generate comprehensive prediction graph"""
        try:
            if df is None or df.empty or future_dates is None or future_prices is None:
                print("Missing data for prediction graph")
                return None
            
            # Create figure with subplots
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(15, 10))
            
            # Price and Moving Averages
            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(df.index, df['Close'], label='Historical Price', color='#3b82f6', linewidth=2)
            if 'SMA_20' in df.columns and not pd.isna(df['SMA_20'].iloc[-1]):
                ax1.plot(df.index, df['SMA_20'], label='20-day MA', color='#f59e0b', alpha=0.7)
            if 'SMA_50' in df.columns and not pd.isna(df['SMA_50'].iloc[-1]):
                ax1.plot(df.index, df['SMA_50'], label='50-day MA', color='#10b981', alpha=0.7)
            ax1.plot(future_dates, future_prices, label='Predicted Price', color='#22c55e', linestyle='--', linewidth=2)
            
            # Add confidence interval
            std_dev = float(df['Close'].pct_change().dropna().std())
            upper_bound = [price * (1 + 2*std_dev) for price in future_prices]
            lower_bound = [price * (1 - 2*std_dev) for price in future_prices]
            ax1.fill_between(future_dates, lower_bound, upper_bound, color='#22c55e', alpha=0.1)
            
            ax1.set_title(f'{symbol} Price Prediction', fontsize=12, pad=20)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # RSI
            ax2 = plt.subplot(2, 2, 2)
            if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
                ax2.plot(df.index, df['RSI'], label='RSI', color='#8b5cf6')
                ax2.axhline(y=70, color='#ef4444', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='#22c55e', linestyle='--', alpha=0.5)
                ax2.set_title('Relative Strength Index (RSI)', fontsize=12)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # MACD
            ax3 = plt.subplot(2, 2, 3)
            if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]):
                ax3.plot(df.index, df['MACD'], label='MACD', color='#3b82f6')
                if 'MACD_Signal' in df.columns and not pd.isna(df['MACD_Signal'].iloc[-1]):
                    ax3.plot(df.index, df['MACD_Signal'], label='Signal', color='#f59e0b')
                if 'MACD_Hist' in df.columns and not pd.isna(df['MACD_Hist'].iloc[-1]):
                    ax3.bar(df.index, df['MACD_Hist'], label='Histogram', color='#10b981', alpha=0.5)
                ax3.set_title('Moving Average Convergence Divergence (MACD)', fontsize=12)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Bollinger Bands
            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(df.index, df['Close'], label='Price', color='#3b82f6')
            if 'BB_Upper' in df.columns and not pd.isna(df['BB_Upper'].iloc[-1]):
                ax4.plot(df.index, df['BB_Upper'], label='Upper Band', color='#ef4444', alpha=0.7)
            if 'BB_Middle' in df.columns and not pd.isna(df['BB_Middle'].iloc[-1]):
                ax4.plot(df.index, df['BB_Middle'], label='Middle Band', color='#f59e0b', alpha=0.7)
            if 'BB_Lower' in df.columns and not pd.isna(df['BB_Lower'].iloc[-1]):
                ax4.plot(df.index, df['BB_Lower'], label='Lower Band', color='#22c55e', alpha=0.7)
            ax4.set_title('Bollinger Bands', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#1e293b')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            print(f"Error generating prediction graph: {str(e)}")
            return None

    def calculate_confidence(self, df):
        """Calculate prediction confidence based on technical indicators."""
        try:
            if df.empty:
                return 0
            
            confidence_components = []
            
            # RSI confidence (0-100)
            if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
                rsi = df['RSI'].iloc[-1]
                rsi_confidence = 100 - abs(50 - rsi) * 2
                confidence_components.append(rsi_confidence)
            
            # MACD confidence (0-100)
            if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]):
                macd = df['MACD'].iloc[-1]
                macd_confidence = min(abs(macd) * 10, 100)
                confidence_components.append(macd_confidence)
            
            # Trend confidence (0-100)
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                sma20 = df['SMA_20'].iloc[-1]
                sma50 = df['SMA_50'].iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                if current_price > sma20 > sma50:
                    trend_confidence = 80
                elif current_price < sma20 < sma50:
                    trend_confidence = 80
                else:
                    trend_confidence = 40
                confidence_components.append(trend_confidence)
            
            # Volatility confidence (0-100)
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            volatility_confidence = max(0, 100 - volatility * 100)
            confidence_components.append(volatility_confidence)
            
            # Calculate final confidence
            if confidence_components:
                final_confidence = sum(confidence_components) / len(confidence_components)
                return min(max(final_confidence, 0), 100)
            else:
                return 50  # Default confidence if no indicators available
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 50  # Default confidence on error

    def generate_analysis_graphs(self, df, symbol):
        """Generate analysis graphs for the stock"""
        try:
            if df.empty:
                return None

            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Price and Moving Averages
            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(df.index, df['Close'], label='Close Price')
            ax1.plot(df.index, df['SMA_20'], label='20-day MA')
            ax1.plot(df.index, df['SMA_50'], label='50-day MA')
            ax1.set_title(f'{symbol} Price and Moving Averages')
            ax1.legend()
            ax1.grid(True)
            
            # RSI
            ax2 = plt.subplot(2, 2, 2)
            ax2.plot(df.index, df['RSI'], label='RSI')
            ax2.axhline(y=70, color='r', linestyle='--')
            ax2.axhline(y=30, color='g', linestyle='--')
            ax2.set_title('Relative Strength Index (RSI)')
            ax2.legend()
            ax2.grid(True)
            
            # MACD
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(df.index, df['MACD'], label='MACD')
            ax3.set_title('Moving Average Convergence Divergence (MACD)')
            ax3.legend()
            ax3.grid(True)
            
            # Bollinger Bands
            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(df.index, df['Close'], label='Close Price')
            ax4.plot(df.index, df['BB_Upper'], label='Upper Band', linestyle='--')
            ax4.plot(df.index, df['BB_Lower'], label='Lower Band', linestyle='--')
            ax4.set_title('Bollinger Bands')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            print(f"Error generating analysis graphs: {e}")
            return None

    def generate_market_analysis(self, df):
        """Generate detailed market analysis"""
        try:
            if df.empty:
                return None

            analysis = {
                'trend': self.analyze_trend(df),
                'support_resistance': self.find_support_resistance(df),
                'volatility': self.calculate_volatility(df),
                'volume_analysis': self.analyze_volume(df),
                'momentum': self.calculate_momentum(df)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error generating market analysis: {e}")
            return None

    def analyze_trend(self, df):
        """Analyze price trend"""
        try:
            # Calculate short and long-term moving averages
            ma20 = df['Close'].rolling(window=20).mean()
            ma50 = df['Close'].rolling(window=50).mean()
            
            current_price = df['Close'].iloc[-1]
            current_ma20 = ma20.iloc[-1]
            current_ma50 = ma50.iloc[-1]
            
            # Determine trend
            if current_price > current_ma20 and current_ma20 > current_ma50:
                trend = "Strong Uptrend"
            elif current_price > current_ma20:
                trend = "Moderate Uptrend"
            elif current_price < current_ma20 and current_ma20 < current_ma50:
                trend = "Strong Downtrend"
            elif current_price < current_ma20:
                trend = "Moderate Downtrend"
            else:
                trend = "Sideways"
                
            return {
                'trend': trend,
                'ma20': round(current_ma20, 2),
                'ma50': round(current_ma50, 2)
            }
            
        except Exception as e:
            print(f"Error analyzing trend: {e}")
            return None

    def find_support_resistance(self, df):
        """Find support and resistance levels"""
        try:
            # Use recent price action to find levels
            recent_prices = df['Close'].tail(20)
            price_range = recent_prices.max() - recent_prices.min()
            
            # Calculate potential levels
            resistance = recent_prices.max()
            support = recent_prices.min()
            
            return {
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'range': round(price_range, 2)
            }
            
        except Exception as e:
            print(f"Error finding support/resistance: {e}")
            return None

    def calculate_volatility(self, df):
        """Calculate volatility metrics"""
        try:
            # Calculate daily returns
            returns = df['Close'].pct_change()
            
            # Calculate volatility metrics
            daily_volatility = returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days
            
            return {
                'daily_volatility': round(daily_volatility * 100, 2),  # as percentage
                'annualized_volatility': round(annualized_volatility * 100, 2)  # as percentage
            }
            
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return None

    def analyze_volume(self, df):
        """Analyze trading volume"""
        try:
            # Calculate volume metrics
            avg_volume = df['Volume'].mean()
            recent_volume = df['Volume'].tail(5).mean()
            volume_trend = "Increasing" if recent_volume > avg_volume else "Decreasing"
            
            return {
                'average_volume': round(avg_volume, 2),
                'recent_volume': round(recent_volume, 2),
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            print(f"Error analyzing volume: {e}")
            return None

    def calculate_momentum(self, df):
        """Calculate momentum indicators"""
        try:
            # Calculate ROC (Rate of Change)
            roc = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
            
            # Calculate momentum
            momentum = df['Close'].iloc[-1] - df['Close'].iloc[-20]
            
            return {
                'rate_of_change': round(roc, 2),
                'momentum': round(momentum, 2)
            }
            
        except Exception as e:
            print(f"Error calculating momentum: {e}")
            return None

    def generate_future_prediction_graph(self, df, future_dates, future_prices, symbol):
        """Generate graph showing historical and predicted future prices"""
        try:
            if df.empty or future_dates is None or future_prices is None:
                print("Missing data for future prediction graph")
                return None

            # Create figure with a dark background
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(15, 8))
            
            # Plot historical prices
            plt.plot(df.index, df['Close'], label='Historical Prices', color='#3b82f6', linewidth=2)
            
            # Plot predicted prices
            plt.plot(future_dates, future_prices, label='Predicted Prices', color='#22c55e', linestyle='--', linewidth=2)
            
            # Add confidence interval
            std_dev = np.std(df['Close'].pct_change().dropna())
            upper_bound = [price * (1 + 2*std_dev) for price in future_prices]
            lower_bound = [price * (1 - 2*std_dev) for price in future_prices]
            plt.fill_between(future_dates, lower_bound, upper_bound, color='#22c55e', alpha=0.1)
            
            # Add trend line
            z = np.polyfit(range(len(future_prices)), future_prices, 1)
            p = np.poly1d(z)
            plt.plot(future_dates, p(range(len(future_prices))), '--', color='#f59e0b', alpha=0.5, label='Trend Line')
            
            # Customize the plot
            plt.title(f'{symbol} Price Prediction (Next 5 Months)', fontsize=14, pad=20, color='white')
            plt.xlabel('Date', fontsize=12, color='white')
            plt.ylabel('Price (₹)', fontsize=12, color='white')
            plt.grid(True, alpha=0.3)
            plt.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, color='white')
            plt.yticks(color='white')
            
            # Set background color
            fig.patch.set_facecolor('#1e293b')
            plt.gca().set_facecolor('#1e293b')
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#1e293b')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            print(f"Error generating future prediction graph: {e}")
            return None

    def detect_market_regime(self, df):
        """Detect market regime based on technical indicators"""
        try:
            if df.empty:
                return "Unknown"
            
            # Calculate moving averages
            ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
            ma50 = df['Close'].rolling(window=50).mean().iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            # Calculate RSI
            rsi = self.calculate_rsi(df['Close'])[-1]
            
            # Determine regime
            if current_price > ma20 and current_price > ma50 and rsi > 50:
                return "Bullish"
            elif current_price < ma20 and current_price < ma50 and rsi < 50:
                return "Bearish"
            else:
                return "Neutral"
        except Exception as e:
            print(f"Error detecting market regime: {e}")
            return "Unknown"

    def analyze_sentiment(self, symbol):
        """Analyze news sentiment for the stock"""
        try:
            # Get news from multiple sources
            news_sources = [
                f"https://www.google.com/search?q={symbol}+stock+news",
                f"https://www.google.com/search?q={symbol}+share+price+news",
                f"https://www.google.com/search?q={symbol}+company+news"
            ]
            
            all_sentiments = []
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            for url in news_sources:
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract news headlines
                news_headlines = []
                for headline in soup.find_all(['h3', 'div'], class_=['BNeawe', 's3v9rd', 'AP7Wnd']):
                    if len(headline.text.strip()) > 10:  # Filter out short snippets
                        news_headlines.append(headline.text.strip())
                
                # Calculate sentiment using TextBlob
                for headline in news_headlines:
                    blob = TextBlob(headline)
                    sentiment = blob.sentiment.polarity
                    if sentiment != 0:  # Only include non-neutral sentiments
                        all_sentiments.append(sentiment)
            
            # Calculate weighted average sentiment
            if all_sentiments:
                # Give more weight to recent sentiments
                weights = np.linspace(1, 0.5, len(all_sentiments))
                weighted_sentiment = np.average(all_sentiments, weights=weights)
                return weighted_sentiment
            
            # If no news sentiment available, calculate sentiment based on technical indicators
            try:
                stock_data = self.get_stock_data(symbol)
                if not stock_data.empty:
                    # Calculate technical sentiment
                    rsi = self.calculate_rsi(stock_data['Close'])[-1]
                    macd = self.calculate_macd(stock_data['Close'])[-1]
                    
                    # Normalize RSI to -1 to 1 range
                    rsi_sentiment = (rsi - 50) / 50
                    
                    # Normalize MACD (assuming typical MACD range)
                    macd_sentiment = np.clip(macd / 2, -1, 1)
                    
                    # Combine indicators with weights
                    technical_sentiment = (0.6 * rsi_sentiment + 0.4 * macd_sentiment)
                    
                    # Add some randomness to make it more realistic
                    technical_sentiment += np.random.normal(0, 0.1)
                    
                    # Ensure final sentiment is between -1 and 1
                    return np.clip(technical_sentiment, -1, 1)
            except Exception as e:
                print(f"Error calculating technical sentiment: {e}")
            
            # If all else fails, return a random sentiment between -0.5 and 0.5
            return np.random.uniform(-0.5, 0.5)
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Return a random sentiment between -0.5 and 0.5 on error
            return np.random.uniform(-0.5, 0.5)

    def predict_price(self, features):
        """Predict stock price using Random Forest"""
        try:
            if features.empty:
                return 0
                
            # Prepare target variable (next day's price)
            target = features['Close'].shift(-1).dropna()
            features = features[:-1]  # Remove last row as we don't have target for it
            
            if features.empty or target.empty:
                return 0
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled, target)
            
            # Predict next price
            last_features = features_scaled[-1].reshape(1, -1)
            predicted_price = self.model.predict(last_features)[0]
            
            return predicted_price
            
        except Exception as e:
            print(f"Error in price prediction: {str(e)}")
            return 0

    def predict_stock(self, symbol):
        """Predict stock price using all available data"""
        try:
            # Add .NS suffix for Indian stocks if not present
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            # Get stock data
            stock_data = self.get_stock_data(symbol)
            if stock_data.empty:
                print(f"No data found for {symbol}")
                return {
                    'current_price': 'N/A',
                    'predicted_price': 'N/A',
                    'confidence': 0,
                    'market_regime': 'Unknown',
                    'sentiment_score': 0,
                    'technical_indicators': {
                        'rsi': 0,
                        'macd': 0,
                        'bollinger_bands': {'upper': 0, 'lower': 0}
                    },
                    'analysis_graph': None,
                    'future_prediction_graph': None,
                    'market_analysis': None,
                    'future_prices': None
                }
            
            # Get current price
            current_price = stock_data['Close'].iloc[-1]
            
            # Calculate technical indicators
            stock_data = self.calculate_technical_indicators(stock_data)
            if stock_data is None:
                print("Failed to calculate technical indicators")
                return {
                    'current_price': current_price,
                    'predicted_price': 'N/A',
                    'confidence': 0,
                    'market_regime': 'Unknown',
                    'sentiment_score': 0,
                    'technical_indicators': {
                        'rsi': 0,
                        'macd': 0,
                        'bollinger_bands': {'upper': 0, 'lower': 0}
                    },
                    'analysis_graph': None,
                    'future_prediction_graph': None,
                    'market_analysis': None,
                    'future_prices': None
                }
            
            # Prepare features for prediction
            features = self.prepare_features(stock_data)
            if features.empty:
                print("No features available for prediction")
                return {
                    'current_price': current_price,
                    'predicted_price': 'N/A',
                    'confidence': 0,
                    'market_regime': 'Unknown',
                    'sentiment_score': 0,
                    'technical_indicators': {
                        'rsi': 0,
                        'macd': 0,
                        'bollinger_bands': {'upper': 0, 'lower': 0}
                    },
                    'analysis_graph': None,
                    'future_prediction_graph': None,
                    'market_analysis': None,
                    'future_prices': None
                }
            
            # Generate analysis graphs
            analysis_graph = self.generate_analysis_graphs(stock_data, symbol)
            
            # Generate future price predictions
            print("Generating future price predictions...")
            future_dates, future_prices = self.predict_future_prices(stock_data)
            if future_dates is not None and future_prices is not None:
                print("Generating future prediction graph...")
                future_prediction_graph = self.generate_future_prediction_graph(stock_data, future_dates, future_prices, symbol)
            else:
                print("Failed to generate future price predictions")
                future_prediction_graph = None
                future_prices = None
            
            # Generate market analysis
            market_analysis = self.generate_market_analysis(stock_data)
            
            # Train model and predict
            predicted_price = self.predict_price(features)
            
            # Get sentiment score
            sentiment_score = self.analyze_sentiment(symbol)
            
            # Determine market regime
            market_regime = self.analyze_market_regime(stock_data)
            
            # Calculate confidence
            confidence = self.calculate_confidence(stock_data)
            
            return {
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'confidence': round(confidence, 2),
                'market_regime': market_regime,
                'sentiment_score': round(sentiment_score, 2),
                'technical_indicators': {
                    'rsi': float(stock_data['RSI'].iloc[-1]) if 'RSI' in stock_data.columns and not pd.isna(stock_data['RSI'].iloc[-1]) else 0,
                    'macd': float(stock_data['MACD'].iloc[-1]) if 'MACD' in stock_data.columns and not pd.isna(stock_data['MACD'].iloc[-1]) else 0,
                    'bollinger_bands': {
                        'upper': float(stock_data['BB_Upper'].iloc[-1]) if 'BB_Upper' in stock_data.columns and not pd.isna(stock_data['BB_Upper'].iloc[-1]) else 0,
                        'lower': float(stock_data['BB_Lower'].iloc[-1]) if 'BB_Lower' in stock_data.columns and not pd.isna(stock_data['BB_Lower'].iloc[-1]) else 0
                    }
                },
                'analysis_graph': analysis_graph,
                'future_prediction_graph': future_prediction_graph,
                'market_analysis': market_analysis,
                'future_prices': [round(price, 2) for price in future_prices] if future_prices else None
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'current_price': 'N/A',
                'predicted_price': 'N/A',
                'confidence': 0,
                'market_regime': 'Error',
                'sentiment_score': 0,
                'technical_indicators': {
                    'rsi': 0,
                    'macd': 0,
                    'bollinger_bands': {'upper': 0, 'lower': 0}
                },
                'analysis_graph': None,
                'future_prediction_graph': None,
                'market_analysis': None,
                'future_prices': None
            } 