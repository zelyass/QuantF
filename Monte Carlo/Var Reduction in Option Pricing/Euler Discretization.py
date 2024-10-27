import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Parameters for the SDE
lambda_ = 2.0  # Drift term
mu = 1.0       # Volatility term
X0 = 1.0       # Initial condition
T = 1.0        # Time horizon

# Function to generate the exact solution at all time points
def exact_solution(lambda_, mu, X0, W, N):
    time_points = np.linspace(0, T, N + 1)
    return X0 * np.exp((lambda_ - 0.5 * mu**2) * time_points + mu * W)

# Function for Euler-Maruyama discretization
def euler_maruyama(lambda_, mu, X0, T, N):
    dt = T / N
    X = np.zeros(N + 1)
    X[0] = X0
    W = np.random.randn(N) * np.sqrt(dt)  # Brownian increments
    W_cumsum = np.concatenate(([0], W.cumsum()))  # Include W(0) = 0
    for j in range(N):
        X[j + 1] = X[j] + lambda_ * X[j] * dt + mu * X[j] * W[j]
    return X, W_cumsum  # Return cumulative sum for W(t)

# Time step sizes
time_steps = [4 * 2**(-8), 2 * 2**(-8), 2**(-8)]
errors = []

# Simulate for each time step size
for dt in time_steps:
    N = int(T / dt)  # Number of steps
    # Euler-Maruyama simulation
    X_euler, W_T = euler_maruyama(lambda_, mu, X0, T, N)
    # Exact solution at all time points using the same num of time points as Euler
    X_exact = exact_solution(lambda_, mu, X0, W_T, N)
    # Error at the endpoint
    error = np.abs(X_exact[-1] - X_euler[-1])
    errors.append(error)
    
    # Create a time vector for plotting
    t = np.linspace(0, T, N + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(t, X_euler, label=f'Euler Approximation (dt = {dt:.5f})')
    plt.plot(t, X_exact, color='red', linestyle='--', label='Exact Solution')
    plt.title(f'Euler-Maruyama Approximation vs Exact Solution (dt = {dt:.5f})')
    plt.xlabel('Time')
    plt.ylabel('X(t)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Print the errors at the endpoint for different time steps
for dt, error in zip(time_steps, errors):
    print(f"Error at the endpoint with dt = {dt:.10f}: {error:.10f}")
