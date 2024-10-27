import numpy as np
import pandas as pd

def S_path(S0, r, sigma, T, n):
    # Time increment
    dt = T / n
    # Initialize price array with zeros and set the first element to S0
    S = np.zeros(n + 1)
    S[0] = S0

    # Generate the price path
    for i in range(1, n + 1):
        Z = np.random.normal()  # Standard normal random variable
        X = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        S[i] = S[i - 1] * np.exp(X)
    return S

# Parameters
S0 = 500      # Initial stock price
r = 0.0175    # Risk-free interest rate
sigma = 0.25  # Volatility
T = 1         # Time to maturity
n = 52        # Number of time steps (weekly)
K = S0       # Strike price
Ns = [10**4, 10**5]  # Different Monte Carlo sample sizes

def Arith_Avg_Crude_MC(S0, r, sigma, T, n, K, N):
    np.random.seed(42)  # Set seed for reproducibility
    temp = 0
    temp2 = 0
    
    for _ in range(N):
        # Generate a price path
        S = S_path(S0, r, sigma, T, n)
        # Calculate the arithmetic average of the path
        Aavg = sum(S[1:]) / n  # Sum from S[1] to S[n] and divide by n
        # Calculate the payoff for the put option
        if Aavg < K:
            disc_payoff = np.exp(-r * T) * (K - Aavg)
            temp += disc_payoff
            temp2 += disc_payoff ** 2
    
    # Monte Carlo estimates
    muhat = temp / N
    s2 = (temp2 / (N - 1)) - (N / (N - 1)) * muhat ** 2
    shat = np.sqrt(s2)
    MSE = s2 / N
    # 95% confidence interval
    ci_error = 1.96 * shat / np.sqrt(N)
    ci_l = muhat - ci_error
    ci_u = muhat + ci_error
    CI = (ci_l, ci_u)
    
    return {
        "N": N,
        "Estimated Price": muhat,
        "MSE": MSE,
        "Confidence Interval": CI
    }

# Run simulations for different values of N
results = [Arith_Avg_Crude_MC(S0, r, sigma, T, n, K, N) for N in Ns]

# Convert results to DataFrame
df_results = pd.DataFrame(results)
df_results.columns = ["Sample Size (N)", "Estimated Price", "MSE", "Confidence Interval"]

# Set pandas display option to show all columns
pd.set_option("display.max_columns", None)
print(df_results)
