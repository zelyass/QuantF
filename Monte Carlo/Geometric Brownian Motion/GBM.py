from scipy.stats import lognorm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

# Set a random seed for reproducibility
np.random.seed(42)

# Set style for the plots
sns.set(style="whitegrid")

# Function to generate geometric Brownian motion distribution
def geometric_brownian_motion(initial_value, drift, volatility, time_period):
    expected_log = np.log(initial_value) + (drift - 0.5 * volatility**2) * time_period
    adjusted_volatility = volatility * np.sqrt(time_period)
    gbm_distribution = lognorm(s=adjusted_volatility, scale=np.exp(expected_log))
    return gbm_distribution

# Plotting for different drift values
fig, ax = plt.subplots(figsize=(12, 6))
drift_values = [-0.2, -0.1, 0, 0.1, 0.3]
colors = sns.color_palette("husl", len(drift_values))

for drift, color in zip(drift_values, colors):
    distribution = geometric_brownian_motion(initial_value=1.0, drift=drift, volatility=0.15, time_period=1)
    x_values = np.linspace(0, distribution.ppf(0.999), 100)
    ax.plot(x_values, distribution.pdf(x_values), color=color, lw=2, alpha=0.85, label=f'Drift ($\mu$)={drift:.2f}')

ax.set_title('Density Function of $S_1$ with Varying Drift Values', fontsize=14)
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()

# Plotting for different volatility values
fig, ax = plt.subplots(figsize=(12, 6))
volatility_values = [0.05, 0.1, 0.2, 0.4]
colors = sns.color_palette("cubehelix", len(volatility_values))

for volatility, color in zip(volatility_values, colors):
    distribution = geometric_brownian_motion(initial_value=1.0, drift=0.1, volatility=volatility, time_period=1)
    x_values = np.linspace(0, distribution.ppf(0.999), 100)
    ax.plot(x_values, distribution.pdf(x_values), color=color, lw=2, alpha=0.85, label=f'Volatility ($\sigma$)={volatility:.2f}')

ax.set_title('Density Function of $S_1$ with Varying Volatility', fontsize=14)
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()

# Plotting for different time periods
fig, ax = plt.subplots(figsize=(12, 6))
time_periods = [0.1, 0.5, 1, 2, 5]
colors = sns.color_palette("muted", len(time_periods))

for time, color in zip(time_periods, colors):
    distribution = geometric_brownian_motion(initial_value=1.0, drift=0.1, volatility=0.2, time_period=time)
    x_values = np.linspace(0, distribution.ppf(0.999), 100)
    ax.plot(x_values, distribution.pdf(x_values), color=color, lw=2, alpha=0.85, label=f'Time ($t$)={time:.1f}')

ax.set_title('Density Function of $S_t$ with Varying Time Periods', fontsize=14)
ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()

# Function to draw mean and variance of the modified GBM
def draw_gbm_mean_variance(initial_price, drift_rate, volatility, time_steps=100):
    fig, (mean_ax, variance_ax) = plt.subplots(1, 2, figsize=(12, 5))
    time_values = np.linspace(0, time_steps, time_steps)
    
    # Plotting Expected Value
    mean_ax.plot(time_values, np.exp(drift_rate * time_values), lw=1.5, color='black', label='Expected Value')
    mean_ax.set_title('Expected Value of $S_t$', fontsize=14)
    mean_ax.set_xlabel('Time ($t$)', fontsize=12)
    mean_ax.set_ylabel('Value', fontsize=12)
    mean_ax.legend()
    mean_ax.set_ylim(0, np.exp(drift_rate * time_steps) * 1.1)  # Adjusting y-limits for clarity

    # Plotting Variance
    variance_ax.plot(time_values, (initial_price**2) * np.exp(2 * drift_rate * time_values) * (np.exp(time_values * volatility**2) - 1), lw=1.5, color='red', label='Variance')
    variance_ax.set_title('Variance of $S_t$', fontsize=14)
    variance_ax.set_xlabel('Time ($t$)', fontsize=12)
    variance_ax.set_ylabel('Variance', fontsize=12)
    variance_ax.legend()
    variance_ax.set_ylim(0, (initial_price**2) * np.exp(2 * drift_rate * time_steps) * (np.exp(time_steps * volatility**2) - 1) * 1.1)  # Adjusting y-limits for clarity

    fig.suptitle(f'Expected Value and Variance of $S_t$ with $S_0$={initial_price:.2f}, $\mu$={drift_rate:.2f}, $\sigma$={volatility:.2f}', fontsize=12)
    plt.tight_layout()
    plt.show()

# Example 1
draw_gbm_mean_variance(initial_price=1.0, drift_rate=0.1, volatility=0.2, time_steps=100)

# Example 2
draw_gbm_mean_variance(initial_price=1.0, drift_rate=-0.05, volatility=0.15, time_steps=100)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_time_series(start=0.0, end=1.0, num_steps=30):
    """Generate a series of time points from start to end."""
    time_interval = (end - start) / num_steps
    time_points = np.arange(start, end + time_interval, time_interval)
    return time_points

def simulate_brownian_motion(time_points, initial_value=0):
    """Simulate a Brownian motion path."""
    num_points = len(time_points)
    time_delta = (time_points[-1] - time_points[0]) / num_points
    increments = norm.rvs(loc=0, scale=np.sqrt(time_delta), size=num_points - 1)
    increments = np.insert(increments, 0, initial_value)
    brownian_path = increments.cumsum()

    return brownian_path

# Generate time series
time_series = generate_time_series(start=1, end=10, num_steps=100)
brownian_motion = simulate_brownian_motion(time_series)

# Parameters for the Geometric Brownian Motion
initial_price = 1
drift = 0.2
volatility = 0.25

# Calculate the Geometric Brownian Motion path
gbm_path = initial_price * np.exp((drift - 0.5 * volatility**2) * time_series + volatility * brownian_motion)

# Plotting the Geometric Brownian Motion path
plt.figure(figsize=(10, 6))
plt.plot(time_series, gbm_path, '-', lw=1.5, label='Geometric Brownian Motion')
plt.title('Geometric Brownian Motion Path', fontsize=16)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
