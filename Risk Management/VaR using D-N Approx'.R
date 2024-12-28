# Delta Normal Approximation to VAR for a Long Option Position using Importance Sampling

# Parameters
s0 <- 100
K <- 100
r <- 0.01
sigma <- 0.5
T <- 1
Dt <- 5/365  # Calculating 5-day VAR

# Function to calculate delta_t
delta_t <- function(s0, K, r, sigma, T, t) {
  d1 <- (log(s0/K) + (r + 0.5*sigma^2)*(T - t)) / (sigma * sqrt(T - t))
  d2 <- d1 - sigma * sqrt(T - t)
  N1 <- pnorm(d1)
  N2 <- pnorm(d2)
  Delta_t = s0 * dnorm(d1) / (sigma * sqrt(T - t)) + s0 * N1 - K * exp(-r * (T - t)) * dnorm(d2) / (sigma * sqrt(T - t))
  return(Delta_t)
}

# Calculating delta_t at t = 0
t = 0
Delta_t = delta_t(s0, K, r, sigma, T, t)

# VAR for a long position in a European call option under Black-Scholes Model
alpha = 0.05
z_alpha = qnorm(1 - alpha)

VAR_L = -z_alpha * sigma * sqrt(Dt) * Delta_t

# Black-Scholes formula to calculate option prices
BScall <- function(t, T, S, K, r, sigma) {
  d1 <- (log(S / K) + (r + 0.5 * sigma^2) * (T - t)) / (sigma * sqrt(T - t))
  d2 <- d1 - sigma * sqrt(T - t)
  N1 <- pnorm(d1)
  N2 <- pnorm(d2)
  y <- S * N1 - K * exp(-r * (T - t)) * N2
  return(y)
}

# Importance Sampling: Change of measure
# Modify the drift of the process to focus on the tail (Increase volatility to capture extreme movements)

# For importance sampling, we'll shift the drift upwards so that extreme moves are more probable
shift_factor = 2  # This can be adjusted based on your needs
mu_is <- r - 0.5 * sigma^2 + shift_factor * sigma  # Adjusted drift for importance sampling

# Simulate the number of VAR breaks using Importance Sampling
N <- 10^6
Breaks <- 0

set.seed(461)

for (i in 1:N) {
  # Simulating using the importance sampling distribution
  S_is <- s0 * exp((mu_is - 0.5 * sigma^2) * Dt + sigma * sqrt(Dt) * rnorm(1))  # Importance sampling path
  V0 <- BScall(0, T, s0, K, r, sigma)
  V <- BScall(Dt, T, S_is, K, r, sigma)
  dV <- V - V0
  
  # Re-weighting the probability based on the ratio of the original and the modified distribution
  weight <- exp(-(mu_is - r) * Dt)  # Importance sampling weight
  if (dV < VAR_L) {
    Breaks <- Breaks + weight
  }
}

# Estimating the alpha (probability of VAR break)
alpha_est_is <- Breaks / N

cat("5-days VaR with the Delta-Normal Approximation:", VAR_L, "\n")
# Output the estimated alpha from importance sampling
cat("Estimated probability of a VAR break using Importance Sampling:", alpha_est_is, "\n")
