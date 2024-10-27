import numpy as np
import pandas as pd
from scipy.stats import norm

def arith_geo_avgs(S0, r, sigma, T, n):
    Dt = T / n
    S = S0
    temp_sum = 0
    temp_prod = 1
    
    for _ in range(n):
        Z = np.random.normal()
        X = (r - 0.5 * sigma ** 2) * Dt + sigma * np.sqrt(Dt) * Z
        S *= np.exp(X)
        temp_sum += S
        temp_prod *= S

    arith_mean = temp_sum / n
    geo_mean = temp_prod ** (1 / n)
    
    return arith_mean, geo_mean

def geo_avg_put(S0, r, sigma, T, n, K):
    """
    Calculates the time-zero price of a geometric average Asian put option.
    """
    # Mean and variance for the log of the geometric mean based on provided context
    mean_geo = (r - 0.5 * sigma ** 2) * (n + 1) / (2 * n)
    variance_geo = (sigma ** 2) * ((n + 1) * (2 * n + 1)) / (6 * n ** 2)
    
    mu_hat = mean_geo + 0.5 * variance_geo
    sigma_hat = np.sqrt(variance_geo)
    
    d1_hat = (np.log(S0 / K) + (mu_hat + 0.5 * sigma_hat ** 2) * T) / (sigma_hat * np.sqrt(T))
    d2_hat = d1_hat - sigma_hat * np.sqrt(T)
    
    # Put option price using the adapted formula
    put_price = np.exp(-r * T) * (K * norm.cdf(-d2_hat) - S0 * np.exp(mu_hat * T) * norm.cdf(-d1_hat))
    
    return put_price

def asian_put_cv(S0, r, sigma, T, n, K, N, p):
    np.random.seed(42)
    m = int(p * N)
    M = N - m
    temp_muA = 0
    temp_muG = 0
    temp_s2G = 0
    disc_AG = 0

    # Pilot run to estimate optimal parameters
    for _ in range(m):
        Aavg, Gavg = arith_geo_avgs(S0, r, sigma, T, n)
        disc_payoffA = np.exp(-r * T) * max(K - Aavg, 0)  # Put option
        disc_payoffG = np.exp(-r * T) * max(K - Gavg, 0)  # Put option
        disc_AG += disc_payoffA * disc_payoffG
        temp_muA += disc_payoffA
        temp_muG += disc_payoffG
        temp_s2G += disc_payoffG ** 2

    muA = temp_muA / m
    muG = temp_muG / m
    s2G = (temp_s2G / (m - 1)) - (m / (m - 1)) * muG ** 2
    chat = (disc_AG - m * muA * muG) / ((m - 1) * s2G)

    # Main CV estimator
    C0_Geo_True = geo_avg_put(S0, r, sigma, T, n, K)
    temp_muCV = 0
    temp_s2CV = 0

    for _ in range(M):
        Aavg, Gavg = arith_geo_avgs(S0, r, sigma, T, n)
        disc_payoffA = np.exp(-r * T) * max(K - Aavg, 0)  # Put option
        disc_payoffG = np.exp(-r * T) * max(K - Gavg, 0)  # Put option
        temp_CV = disc_payoffA - chat * (disc_payoffG - C0_Geo_True)
        temp_muCV += temp_CV
        temp_s2CV += temp_CV ** 2

    muCV = temp_muCV / M
    s2CV = (temp_s2CV / (M - 1)) - (M / (M - 1)) * muCV ** 2
    sCV = np.sqrt(s2CV)
    MSE = s2CV / M
    ci_error = 1.96 * sCV / np.sqrt(M)
    ci_lower = muCV - ci_error
    ci_upper = muCV + ci_error
    CI = (ci_lower, ci_upper)

    return N, muCV, MSE, CI

  
  # Example usage
S0 = 500  # Initial stock price
r = 0.0175  # Risk-free interest rate
sigma = 0.25  # Volatility
T = 1  # Time to maturity
n = 52  # Number of time steps (weekly)
K = S0  # Strike price
N = 10**5

# Get results
result = asian_put_cv(S0, r, sigma, T, n, K, N, 0.5)

# Convert results to a DataFrame
results_df = pd.DataFrame({
    'N': [result[0]],
    'Estimated Value (muCV)': [result[1]],
    'Mean Squared Error (MSE)': [result[2]],
    'Confidence Interval Lower Bound (cil)': [result[3][0]],
    'Confidence Interval Upper Bound (ciu)': [result[3][1]]
})

# Display all columns in the DataFrame
pd.set_option('display.max_columns', None)  # Show all columns
print(results_df)
