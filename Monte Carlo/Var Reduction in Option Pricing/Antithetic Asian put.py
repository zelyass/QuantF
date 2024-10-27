import numpy as np
import pandas as pd

# Parameters
S0 = 500      # Initial stock price
r = 0.0175    # Risk-free interest rate
sigma = 0.25  # Volatility
T = 1         # Time to maturity
n = 52        # Number of time steps (weekly)
K = S0       # Strike price
Ns = [10**4, 10**5]  # Different Monte Carlo sample sizes

# Define functions
def Arith_Avg_Anti_Sums(S0, r, sigma, T, n):
    dt = T / n
    S, Sa = S0, S0
    tempSum, tempSuma = 0, 0
    
    for _ in range(n):
        Z = np.random.normal()
        X = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        Xa = (r - 0.5 * sigma ** 2) * dt - sigma * np.sqrt(dt) * Z
        S *= np.exp(X)
        Sa *= np.exp(Xa)
        tempSum += S
        tempSuma += Sa
        
    return (tempSum / n, tempSuma / n)

def Arith_crude_MC(S0, r, sigma, T, n, K, N):
    if N % 2 != 0:
        raise ValueError("Error: N must be even")
    
    M = N // 2
    np.random.seed(42)  # Set seed for reproducibility
    temp, temp2 = 0, 0
    
    for _ in range(M):
        Aavg1, Aavg2 = Arith_Avg_Anti_Sums(S0, r, sigma, T, n)
        # Calculate discounted payoffs for the put option
        disc_payoff1 = np.exp(-r * T) * max(K - Aavg1, 0)
        disc_payoff2 = np.exp(-r * T) * max(K - Aavg2, 0)
        disc_payoff_avg = (disc_payoff1 + disc_payoff2) / 2
        temp += disc_payoff_avg
        temp2 += disc_payoff_avg ** 2

    # Monte Carlo estimates
    muhat = temp / M
    s2 = (temp2 / (M - 1)) - (M / (M - 1)) * muhat ** 2
    shat = np.sqrt(s2)
    MSE = s2 / M
    # 95% confidence interval
    ci_error = 1.96 * shat / np.sqrt(M)
    ci_l = muhat - ci_error
    ci_u = muhat + ci_error
    CI = (ci_l, ci_u)

    return {
        "Sample Size (N)": N,
        "Estimated Price": muhat,
        "MSE": MSE,
        "Confidence Interval": CI
    }

# Run simulations for different values of N
results = [Arith_crude_MC(S0, r, sigma, T, n, K, N) for N in Ns]

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Set pandas display option to show all columns
pd.set_option("display.max_columns", None)

# Display the DataFrame
print(df_results)
