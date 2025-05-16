from flask import Flask, render_template, request, jsonify, Response, flash, session
import yfinance as yf
import time
import requests
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ai_predictor import StockAI
import json

# Flask is used to connect the backend Python code and the frontend HTML page
# yfinance library is used to get live stock prices and price history for analysis
# Time is used for updating live prices and other time-related operations
# Requests is used for sending HTTP requests to external APIs like Google and News API
# BeautifulSoup is used for web scraping, particularly for extracting data from HTML
# Matplotlib is used for generating graphs and charts for stock analysis

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here'

portfolio = {}  # Dictionary to store user's portfolio
venkat = {}  # Dictionary to store purchased prices of stocks
balance = 100000  # Initial balance for the user

# Initialize the AI predictor
stock_ai = StockAI()

# The chatbot_response function generates a response for the user's input by performing a Google search
# The get_news function fetches the latest business news from News API
# The display_portfolio function generates a summary of the user's portfolio including current values and profit/loss
# The generate_portfolio_pie_chart function creates a pie chart to visualize portfolio performance
# The buy_stock and sell_stock functions handle buying and selling of stocks respectively

def yahoo_search(query):
    try:
        url = f"https://search.yahoo.com/search?p={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers) #It is used to send request to theat url to get the data fro that website
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
         # Find the relevant HTML elements containing the answers
        # Adjust this part according to the structure of the website
        results = soup.find_all('div', class_='algo-sr')       #We get this class after inspecting the yahoo finance page
        return results[0].get_text() if results else "No results found"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"     #we used try and except for error handling because in this big code we are able to see where the error happening



def get_news():
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": "in",
        "category": "business",
        "apiKey": "361ff1f4a3384138b173d1e5d7df3bca"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        return articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return [{"title": "Error fetching news. Please try again later.", "description": "", "url": "#"}]

def display_portfolio(portfolio, balance):
    portfolio_info = {
        'total_value': 0,
        'total_investment': 0,
        'return_percentage': 0,
        'holdings': [],
        'best_performer': {'symbol': 'N/A', 'return_percentage': 0},
        'worst_performer': {'symbol': 'N/A', 'return_percentage': 0},
        'beta': 0,
        'sharpe_ratio': 0
    }
    
    total_investment = 0
    current_portfolio_value = 0
    best_return = float('-inf')
    worst_return = float('inf')
    
    for symbol, shares in portfolio.items():
        try:
            stock = yf.Ticker(symbol + ".NS")
            current_price = stock.history(period="1d")["Close"].iloc[-1]
            current_value = current_price * shares
            purchased_value = venkat[symbol] * shares
            total_investment += purchased_value
            current_portfolio_value += current_value
            
            return_percentage = ((current_value - purchased_value) / purchased_value) * 100
            
            # Update best and worst performers
            if return_percentage > best_return:
                best_return = return_percentage
                portfolio_info['best_performer'] = {'symbol': symbol, 'return_percentage': return_percentage}
            if return_percentage < worst_return:
                worst_return = return_percentage
                portfolio_info['worst_performer'] = {'symbol': symbol, 'return_percentage': return_percentage}
            
            # Add holding information
            portfolio_info['holdings'].append({
                'symbol': symbol,
                'quantity': shares,
                'avg_price': round(venkat[symbol], 2),
                'current_price': round(current_price, 2),
                'value': round(current_value, 2),
                'return_percentage': round(return_percentage, 2),
                'percentage': round((current_value / current_portfolio_value * 100), 2) if current_portfolio_value > 0 else 0
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Calculate portfolio metrics
    portfolio_info['total_value'] = round(current_portfolio_value, 2)
    portfolio_info['total_investment'] = round(total_investment, 2)
    portfolio_info['return_percentage'] = round(((current_portfolio_value - total_investment) / total_investment * 100), 2) if total_investment > 0 else 0
    
    # Generate pie chart for portfolio performance
    generate_portfolio_pie_chart(portfolio)
    
    return portfolio_info


def generate_portfolio_pie_chart(portfolio):
    labels = list(portfolio.keys())
    sizes = list(portfolio.values())

    plt.figure(figsize=(6, 6))  # It indicates the figure size
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal means it ensures the pie chart is circle 
    plt.title('Portfolio Performance')
    plt.savefig('static/portfolio_pie_chart.png')  # Save the pie chart as a PNG file
    plt.close()

def buy_stock(portfolio, symbol, shares):
    global balance
    symbol = symbol.upper()  # Always use uppercase, no .NS in portfolio
    print("[BUY] Before:", portfolio)
    try:
        stock = yf.Ticker(symbol + ".NS")
        price = stock.history(period="1d")["Close"].iloc[-1]
        if symbol in venkat:
            venkat[symbol] = float(f"{price:.2f}")
        else:
            venkat[symbol] = float(f"{price:.2f}")

        total_cost = price * shares
        if total_cost > balance:
            return "Insufficient balance to buy!"
        balance = balance - total_cost
        if symbol in portfolio:
            portfolio[symbol] += shares
        else:
            portfolio[symbol] = shares
        print("[BUY] After:", portfolio)
        return f"Bought {shares} shares of {symbol} at ₹{price:.2f} each.\nRemaining balance: ₹{balance:.2f}"
    except Exception as e:
        print("[BUY] Error:", e)
        return "Error: " + str(e)

def sell_stock(portfolio, symbol, shares):
    global balance
    symbol = symbol.upper()  # Always use uppercase, no .NS in portfolio
    print("[SELL] Before:", portfolio)
    if symbol not in portfolio:
        return "You don't own any shares of " + symbol
    if portfolio[symbol] < shares:
        return "You don't have enough shares to sell!"
    stock = yf.Ticker(symbol + ".NS")
    price = stock.history(period="1d")["Close"].iloc[-1]
    balance = balance + (price * shares)
    portfolio[symbol] -= shares
    print("[SELL] After:", portfolio)
    return f"Sold {shares} shares of {symbol} at ₹{price:.3f} each.\nRemaining balance: ₹{balance:.3f}"

def get_stock_prices(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if not hist.empty and 'Close' in hist.columns:
                v = hist['Close'].iloc[-1]
                data[ticker] = f"{v:.2f}"
            else:
                data[ticker] = "N/A"
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            data[ticker] = "N/A"
    return data



def analyze_stock(symbol):
    # Fetch historical stock data
    stock_data = yf.download(symbol, start="2023-01-01", end="2024-04-21")

    # Calculate moving averages
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

    # Plotting stock data and moving averages
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.plot(stock_data['MA50'], label='50-Day Moving Average')
    plt.plot(stock_data['MA200'], label='200-Day Moving Average')
    plt.title(f'{symbol} Stock Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('static/stock1_analysis.png')  # Save the analysis plot as a PNG file
    plt.close()

    # Calculate returns
    stock_data['Daily Returns'] = stock_data['Close'].pct_change()
    stock_data['Cumulative Returns'] = (1 + stock_data['Daily Returns']).cumprod() - 1

    # Plotting cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Cumulative Returns'], label='Cumulative Returns')
    plt.title(f'{symbol} Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.savefig('static/stock1_returns.png')  # Save the returns plot as a PNG file
    plt.close()
def google_search(query):
    try:
        url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers) #Sends the request to get the information from the google
        response.raise_for_status()  
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd') #We got the class code by inspecting the google search page
        return results[0].get_text() if results else "No results found"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"

def chatbot_response(user_input):
    if user_input.lower() == 'exit':
        return "Goodbye!"
    else:
        return google_search(user_input)
    


# Routes for different pages: index, portfolio, buy, sell, latest_news, chatbot, analyze
# The index route displays the homepage with scrolling live stock prices
# The portfolio route shows the user's portfolio and its performance
# The buy route allows users to buy stocks
# The sell route allows users to sell stocks
# The latest_news route displays the latest business news
# The chatbot route provides a chat interface where users can ask questions
# The analyze route analyzes a stock's performance and displays relevant charts




@app.route('/')
def index():
    # Get prices for some popular Indian stocks
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    stock_prices = get_stock_prices(tickers)
    return render_template('index.html', stock_prices=stock_prices)

@app.route('/portfolio')
def view_portfolio():
    portfolio = json.loads(session.get('portfolio', '{}'))
    portfolio_info = display_portfolio(portfolio, balance)
    return render_template('portfolio.html', portfolio_info=portfolio_info)

@app.route('/buy', methods=['GET', 'POST'])
def buy():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        shares = int(request.form['shares'])
        print(f"[BUY] Attempting to buy {shares} shares of {symbol}")
        portfolio = json.loads(session.get('portfolio', '{}'))
        message = buy_stock(portfolio, symbol, shares)
        session['portfolio'] = json.dumps(portfolio)
        return render_template('buy.html', message=message, symbol=symbol, shares=shares)
    return render_template('buy.html')

@app.route('/confirm_buy', methods=['POST'])
def confirm_buy():
    symbol = request.form['symbol'].upper()
    shares = int(request.form['shares'])
    print(f"[CONFIRM BUY] Attempting to buy {shares} shares of {symbol}")
    portfolio = json.loads(session.get('portfolio', '{}'))
    message = buy_stock(portfolio, symbol, shares)
    session['portfolio'] = json.dumps(portfolio)
    return render_template('message.html', message=message)

@app.route('/sell', methods=['GET', 'POST'])
def sell():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        shares = int(request.form['shares'])
        print(f"[SELL] Attempting to sell {shares} shares of {symbol}")
        portfolio = json.loads(session.get('portfolio', '{}'))
        message = sell_stock(portfolio, symbol, shares)
        session['portfolio'] = json.dumps(portfolio)
        return render_template('sell.html', message=message, symbol=symbol, shares=shares)
    return render_template('sell.html')

@app.route('/confirm_sell', methods=['POST'])
def confirm_sell():
    symbol = request.form['symbol'].upper()
    shares = int(request.form['shares'])
    print(f"[CONFIRM SELL] Attempting to sell {shares} shares of {symbol}")
    portfolio = json.loads(session.get('portfolio', '{}'))
    message = sell_stock(portfolio, symbol, shares)
    session['portfolio'] = json.dumps(portfolio)
    return render_template('message.html', message=message)

@app.route('/update_prices')
def update_prices():
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'TATAMOTORS.NS', 'MRF.NS', 'TCS.NS', "HSCL.NS"]
    stock_prices = get_stock_prices(tickers)
    return jsonify(stock_prices)    #this app route updates the price of tickers for every 10 seconds

@app.route('/stream')
def stream():
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'TATAMOTORS.NS', 'MRF.NS', 'TCS.NS', "HSCL.NS"]
    def event_stream():
        while True:
            yield 'data: {}\n\n'.format(jsonify(get_stock_prices(tickers)))
            time.sleep(10)  # Update after every 10 seconds 
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/latest_news')
def latest_news():
    news = get_news()
    return render_template('stock_news.html', latest_news=news)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.form.get('user_input', '')
        if user_input:
            response = chatbot_response(user_input)
            return render_template("chatbot.html", response=response)
        else:
            return render_template("chatbot.html", response="No input provided")
    else:
        return render_template("chatbot.html", response="")

@app.route('/ai-predict')
def ai_predict():
    return render_template('ai_predict.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        symbol = request.form['symbol'].upper()
        prediction = stock_ai.predict_stock(symbol)
        return render_template('ai_predict.html', prediction=prediction)
    except Exception as e:
        flash(f'Error analyzing stock: {str(e)}', 'error')
        return render_template('ai_predict.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
















