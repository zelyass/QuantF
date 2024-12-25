import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Geometric Brownian Motion simulation
def simulate_gbm(S0, mu, sigma, T, steps, n_simulations):
    dt = T / steps
    simulations = np.zeros((n_simulations, steps + 1))
    simulations[:, 0] = S0
    for t in range(1, steps + 1):
        Z = np.random.normal(size=n_simulations)
        simulations[:, t] = simulations[:, t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return simulations

# Fetch and preprocess data
def fetch_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        stock_data["Log_Return"] = np.log(stock_data["Close"] / stock_data["Close"].shift(1))
        stock_data["Volatility"] = stock_data["Log_Return"].rolling(30).std() * np.sqrt(252)  # Annualized
        stock_data.dropna(inplace=True)
        data[ticker] = stock_data
    return data

# Sequence preparation for LSTM
def create_sequences(data, sequence_length=30):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i:i+sequence_length].values)
        targets.append(data.iloc[i+sequence_length]["Log_Return"])
    return np.array(sequences), np.array(targets)

# Build the LSTM model
def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32))(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Portfolio optimization function
def optimize_portfolio(returns, risk_free_rate=0.02):
    n_assets = returns.shape[0]

    def portfolio_sharpe(weights):
        portfolio_return = np.dot(weights, returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns), weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility  # Negative for minimization

    # Constraints: weights sum to 1, weights >= 0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess: equal weights
    initial_weights = np.ones(n_assets) / n_assets

    # Minimize the negative Sharpe ratio
    result = minimize(portfolio_sharpe, initial_weights, constraints=constraints, bounds=bounds)
    return result.x if result.success else initial_weights

def backtest_portfolio(data, predictions, tickers, simulations, risk_free_rate=0.02, rebalance_frequency=25, transaction_cost=0.0001):
    portfolio_returns, portfolio_volatility, portfolio_weights, sharpe_ratios = [], [], [], []
    reshaped_predictions = predictions[:len(predictions) // len(tickers) * len(tickers)].reshape(-1, len(tickers))
    all_summaries = []  # To collect all the summary DataFrames
    previous_weights = None  # To track weights for transaction cost calculation

    # Rebalance only every rebalance_frequency days
    for i in range(len(reshaped_predictions)):
        if i % rebalance_frequency == 0:  # Rebalance only every rebalance_frequency days
            daily_predictions = reshaped_predictions[i]
            momentum = np.maximum(daily_predictions, 0)
            optimal_weights = momentum / np.sum(momentum) if np.sum(momentum) > 0 else optimize_portfolio(daily_predictions, risk_free_rate)

            # Apply transaction cost if weights have changed (i.e., rebalancing occurred)
            if previous_weights is not None:
                transaction_costs = transaction_cost * np.sum(np.abs(optimal_weights - previous_weights))
            else:
                transaction_costs = 0  # No transaction costs for the initial allocation

            portfolio_weights.append(optimal_weights)
            previous_weights = optimal_weights
        else:
            # Maintain previous weights if not rebalancing
            optimal_weights = portfolio_weights[-1] if portfolio_weights else optimize_portfolio(daily_predictions, risk_free_rate)
            transaction_costs = 0  # No transaction costs on non-rebalancing days

        # Ensure we're not going beyond the simulation steps
        if i < simulations[tickers[0]].shape[1]:
            simulated_returns = [simulations[ticker][:, i].mean() / S0[ticker] for ticker in tickers]
            portfolio_return = np.dot(optimal_weights, simulated_returns) - transaction_costs
            portfolio_returns.append(portfolio_return)
            portfolio_volatility.append(np.std(simulated_returns))

            # Calculate Sharpe Ratio
            portfolio_volatility_current = np.std(simulated_returns)
            sharpe_ratio = (np.mean(simulated_returns) - risk_free_rate) / portfolio_volatility_current if portfolio_volatility_current != 0 else 0
            sharpe_ratios.append(sharpe_ratio)

            # Create a DataFrame to summarize the data for this period
            summary_data = {
                "Time": [i + 1],
                "Portfolio Return": [portfolio_return],
                "Portfolio Volatility": [np.std(simulated_returns)],
                "Sharpe Ratio": [sharpe_ratio]
            }

            # Add the weights of each stock to the summary
            for j, ticker in enumerate(tickers):
                summary_data[f"Weight in {ticker}"] = [optimal_weights[j]]

            # Append the DataFrame to the list
            all_summaries.append(pd.DataFrame(summary_data))

    # Concatenate all the DataFrames into a single DataFrame
    final_summary_df = pd.concat(all_summaries, ignore_index=True)

    return np.array(portfolio_returns), np.array(portfolio_volatility), portfolio_weights, final_summary_df


# Main execution
tickers = ["BSX", "NVDA", "AAPL", "GOOGL", "TSM"]
data = fetch_data(tickers, "2015-04-22", "2024-12-18")

# Preprocess data and prepare LSTM sequences
scaler = MinMaxScaler()
X, y = [], []
for ticker, df in data.items():
    scaled_features = scaler.fit_transform(df[["Log_Return", "Volatility"]])
    scaled_df = pd.DataFrame(scaled_features, columns=["Log_Return", "Volatility"])
    seq_X, seq_y = create_sequences(scaled_df, sequence_length=30)
    X.append(seq_X)
    y.append(seq_y)

X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train LSTM model
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Generate predictions
y_pred = model.predict(X_test)

# GBM simulation
mu, sigma, S0 = {}, {}, {}
simulations = {}
T = 1
steps = 252
n_simulations = 1000

for ticker, df in data.items():
    mu[ticker] = df["Log_Return"].mean()
    sigma[ticker] = df["Volatility"].mean()
    S0[ticker] = df["Close"].iloc[-1]
    simulations[ticker] = simulate_gbm(S0[ticker], mu[ticker], sigma[ticker], T, steps, n_simulations)

# Backtest portfolio
portfolio_returns, portfolio_volatility, portfolio_weights, final_summary_df = backtest_portfolio(
    data, y_pred, tickers, simulations)

# Display the consolidated DataFrame
print(final_summary_df)

# Results visualization
plt.figure(figsize=(14, 8))
plt.plot(portfolio_returns[:200], label="Portfolio Returns")
plt.title("Portfolio Returns Over Time")
plt.xlabel("Time")
plt.ylabel("Portfolio Return")
plt.legend()
plt.show()

# Calculate portfolio value with initial investment 10K
AA = 10000 * portfolio_returns[:200]

# Results visualization
plt.figure(figsize=(14, 8))
plt.plot(AA, label="Portfolio Value")
plt.title("Portfolio Value Across the Period")
plt.xlabel("Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()

plt.figure(figsize=(14, 8))
for i, ticker in enumerate(tickers):
    plt.plot([w[i] for w in portfolio_weights], label=f"Weight in {ticker}")
plt.title("Portfolio Weights Over Time")
plt.xlabel("Rebalance Period")
plt.ylabel("Weight")
plt.legend()
plt.show()
