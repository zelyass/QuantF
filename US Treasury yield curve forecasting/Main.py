import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# ---------------------
# Problem Setup
# ---------------------

T = 1.0             # Maturity
K = 1.0             # Strike price
r = 0.05            # Risk-free rate
sigma_val = 0.2     # Volatility
x0 = np.array([1.0])  # Initial asset price
dim = 1             # Dimension of state

# ---------------------
# Model Coefficients
# ---------------------

def mu(t, x):
    return np.array([r * x[0]])

def sigma(t, x):
    return np.array([[sigma_val * x[0]]])  # Shape (1,1)

def Phi(x):
    return max(x[0] - K, 0)  # Payoff of a European call option

def f(t, x, y, z):
    return -r * y  # Discounting in BSDE

# ---------------------
# Time Discretization
# ---------------------

N = 20
dt = T / N
time_grid = np.linspace(0, T, N + 1)

def phi_n(s):
    """Return the closest previous time grid point φₙ(s)"""
    return max(tk for tk in time_grid if tk <= s)

# ---------------------
# Derivative Approximations
# ---------------------

def finite_diff_grad(v_func, t, x, h=1e-4):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = h
        grad[i] = (v_func(t, x + dx) - v_func(t, x - dx)) / (2 * h)
    return grad

def finite_diff_hess(v_func, t, x, h=1e-4):
    d = len(x)
    hess = np.zeros((d, d))
    for i in range(d):
        dx_i = np.zeros_like(x)
        dx_i[i] = h
        hess[i, i] = (v_func(t, x + dx_i) - 2 * v_func(t, x) + v_func(t, x - dx_i)) / h**2
    return hess

# ---------------------
# Generator Lⁿ Operator
# ---------------------

def generator_Ln(t, x, v_func):
    t_phi = phi_n(t)
    grad = finite_diff_grad(v_func, t, x)
    hess = finite_diff_hess(v_func, t, x)
    mu_val = mu(t_phi, x)
    sigma_val = sigma(t_phi, x)
    term1 = 0.5 * np.trace(sigma_val @ sigma_val.T @ hess)
    term2 = np.dot(mu_val, grad)
    return term1 + term2

# ---------------------
# Simulate Forward SDE
# ---------------------

def simulate_forward(t0, x0, dt, N_steps):
    x = x0.copy()
    t = t0
    path = [x.copy()]
    for _ in range(N_steps):
        mu_val = mu(t, x)
        sigma_val = sigma(t, x)
        dW = np.random.normal(0, np.sqrt(dt), size=(dim,))
        x = x + mu_val * dt + sigma_val @ dW
        path.append(x.copy())
        t += dt
    return np.array(path)

# ---------------------
# Correction Term Ψⁿ
# ---------------------

def compute_correction(t_k, x_k, v_func, M_samples, N_steps):
    total = 0.0
    for _ in range(M_samples):
        path = simulate_forward(t_k, x_k.copy(), dt, N_steps)
        integral = 0.0
        t = t_k
        for i in range(N_steps):
            x = path[i]
            fv = f(t, x, v_func(t, x), finite_diff_grad(v_func, t, x) @ sigma(t, x))
            Lt_v = generator_Ln(t, x, v_func)
            integral += (fv + Lt_v) * dt
            t += dt
        correction = integral + Phi(path[-1]) - v_func(t_k, x_k)
        total += correction
    return total / M_samples

# ---------------------
# Regression Operator Pᵣ
# ---------------------

def fit_regression(grid, values):
    X = np.array([[t] + list(x) for t, x in grid])
    y = np.array(values)
    model = make_pipeline(PolynomialFeatures(3), Ridge(alpha=1e-4))
    model.fit(X, y)
    return lambda t, x: model.predict([[t] + list(x)])[0]

# ---------------------
# Main Algorithm 5
# ---------------------

def run_picard_fbsde(T, N, M_samples, n_points, n_iter):
    dt = T / N
    v_func = lambda t, x: 0.0  # Initial approximation

    for r in range(n_iter):
        grid = [(np.random.uniform(0, T), np.random.uniform(0.5, 1.5, size=(1,))) for _ in range(n_points)]
        targets = []
        for (t_k, x_k) in grid:
            c_k = compute_correction(t_k, x_k, v_func, M_samples, N_steps=N)
            targets.append(v_func(t_k, x_k) + c_k)
        v_func = fit_regression(grid, targets)

    return v_func

# ---------------------
# Run Algorithm
# ---------------------

v_approx = run_picard_fbsde(
    T=T,
    N=N,
    M_samples=100,
    n_points=200,
    n_iter=5
)

# ---------------------
# Plot Approximation at t = 0
# ---------------------

x_vals = np.linspace(0.5, 1.5, 100)
v_vals = [v_approx(0.0, np.array([x])) for x in x_vals]
bs_exact = [max(x - K, 0) * np.exp(-r * T) for x in x_vals]  # true discounted payoff

plt.figure(figsize=(8, 5))
plt.plot(x_vals, v_vals, label="FBSDE Approximation", linewidth=2)
plt.plot(x_vals, bs_exact, '--', label="Discounted Payoff", linewidth=2)
plt.xlabel("Initial Asset Price $x$")
plt.ylabel("Option Price")
plt.title("European Call Option Pricing via Forward Picard Iteration")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------
# Estimate and Variance at (t=0, x0)
# ---------------------

def estimate_with_variance(v_func, x0, M_samples=1000):
    estimates = []
    for _ in range(M_samples):
        path = simulate_forward(0.0, x0.copy(), dt, N)
        payoff = Phi(path[-1])
        integral = 0.0
        t = 0.0
        for i in range(N):
            x = path[i]
            z = finite_diff_grad(v_func, t, x) @ sigma(t, x)
            integrand = f(t, x, v_func(t, x), z) + generator_Ln(t, x, v_func)
            integral += integrand * dt
            t += dt
        estimate = integral + payoff
        estimates.append(estimate)

    estimates = np.array(estimates)
    price_mean = np.mean(estimates)
    price_var = np.var(estimates, ddof=1)
    price_se = np.std(estimates, ddof=1) / np.sqrt(M_samples)

    return price_mean, price_var, price_se

# Compute estimate and variance
price, variance, std_error = estimate_with_variance(v_approx, x0=np.array([1.0]), M_samples=1000)

print(f"Estimated Call Option Price: {price:.6f}")
print(f"Estimated Variance: {variance:.6f}")
print(f"Standard Error: {std_error:.6f}")
