import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the European put option
S0 = 10  # Initial stock price
K = 9    # Strike price
sigma = 0.1  # Volatility
r = 0.06  # Risk-free rate
T = 1    # Time to maturity

# Number of samples to use (logarithmic scale)
sample_sizes = np.logspace(1, 6, num=20, dtype=int)

# Function to calculate the payoff of a European put option
def put_payoff(S_T, K):
    return np.maximum(K - S_T, 0)

# Monte Carlo simulation of the European put option price
def monte_carlo_put_price(S0, K, r, sigma, T, num_samples):
    Z = np.random.randn(num_samples)  # Standard normal samples
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)  # Simulated stock price at T
    payoff = put_payoff(S_T, K)
    discounted_payoff = np.exp(-r * T) * payoff
    return discounted_payoff

# Black-Scholes Formula for a European Put Option
def black_scholes_put(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put_price

# Calculate the true Black-Scholes put option price
true_put_value = black_scholes_put(S0, K, r, sigma, T)

# Store the option values, standard deviations, and confidence intervals
put_values = []
conf_intervals_lower = []
conf_intervals_upper = []

for N in sample_sizes:
    payoffs = monte_carlo_put_price(S0, K, r, sigma, T, N)
    mean_put_price = np.mean(payoffs)  # Sample mean (put option value)
    variance = np.var(payoffs, ddof=1)  # Sample variance
    std_error = np.sqrt(variance / N)  # Sample standard deviation
    conf_interval = 1.96 * std_error  # 95% confidence interval
    
    # Store values for plotting
    put_values.append(mean_put_price)
    conf_intervals_lower.append(mean_put_price - conf_interval)
    conf_intervals_upper.append(mean_put_price + conf_interval)

# Plotting the results
plt.figure(figsize=(8, 6))

# Plot the Monte Carlo estimates with error bars (95% confidence intervals)
plt.errorbar(sample_sizes, put_values,
             yerr=[np.array(put_values) - np.array(conf_intervals_lower),
                   np.array(conf_intervals_upper) - np.array(put_values)],
             fmt='o', color='blue', label='MC put value approx', 
             ecolor='lightblue', elinewidth=2, capsize=3)

# Plot the Black-Scholes true value as a reference
plt.axhline(true_put_value, color='red', linestyle='--', 
             label=f'True put value (BS): {true_put_value:.4f}')

# Logarithmic scale on x-axis
plt.xscale('log')
plt.xlabel('Num samples')
plt.ylabel('Put option value approx')
plt.legend()
plt.title('Monte Carlo Approximation of European Put Option Value with 95% CI')
plt.grid(True)
plt.show()
