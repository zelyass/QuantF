import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
S0 = 100
K = S0 * np.exp(r * T)
sigma = 0.25
r = 0.03
T = 1
NF = 25  # Number of space steps
F_max = 5 * S0  # Assume F_max is five x the initial price for boundary conditions
Delta_F = F_max / NF

# Grid of F and τ
F_grid = np.linspace(0, F_max, NF+1)

# FTCS Scheme Implementation
def FTCS_scheme(N_tau, Delta_tau, Delta_F, sigma, r, F_grid, K):
    C = np.maximum(F_grid - K, 0)  # Initial condition at τ=0 (maturity)
    C_all = np.zeros((N_tau+1, NF+1))  # To store the option prices at all times and space points
    
    # Store the initial option prices at τ=0
    C_all[0, :] = C
    
    for i in range(1, N_tau+1):
        C_new = C.copy()
        for j in range(1, NF):
            F_j = F_grid[j]
            C_new[j] = C[j] + Delta_tau * (0.5 * sigma**2 * j**2 * (C[j+1] - 2*C[j] + C[j-1]) - r * C[j])
        # Boundary conditions
        C_new[0] = 0
        C_new[-1] = F_grid[-1] - K * np.exp(-r * (T - (i) * Delta_tau))
        C_all[i, :] = C_new
        C = C_new
    
    return C_all

# Stability Investigation for Various N_tau Values
def investigate_stability(NF, Delta_F, sigma, r, F_grid, K):
    # Range of N_tau values for stability investigation
    N_tau_values = [5,15,20,25, 26,27,28,29,50,75,80,87,88,89,90,100]  # Number of time steps
    
    for i, N_tau in enumerate(N_tau_values):
        # we adjust Delta_tau according to the number of time steps
        Delta_tau = T / N_tau
        
        #we will compute the option prices for the given N_tau
        option_prices = FTCS_scheme(N_tau, Delta_tau, Delta_F, sigma, r, F_grid, K)
        
        # we the mesh grid for 3D plotting
        tau_values = np.linspace(0, T, N_tau+1)
        F_grid_3d, tau_values_3d = np.meshgrid(F_grid, tau_values)
        
        # Create the 3D plot in a new figure for each N_tau
        fig = plt.figure(figsize=(12, 8), dpi=100)  # Increased size and resolution
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(F_grid_3d, tau_values_3d, option_prices, cmap='viridis')

        # Explicitly set axis limits for clarity
        ax.set_xlim(0, F_max)
        ax.set_ylim(0, T)
        ax.set_zlim(0, np.max(option_prices) * 1.1)  # Leave a margin above the highest value

        # Labels and title
        ax.set_xlabel('Forward Price F', labelpad=10)
        ax.set_ylabel('Time to Maturity τ', labelpad=10)
        ax.set_zlabel('Option Price', labelpad=10)
        ax.set_title(f'Option Price Surface for N_tau = {N_tau}', pad=20)

        # Add a color bar for reference
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

        # Optimize layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Show the plot for this particular N_tau
        plt.show()

# Run the stability investigation
investigate_stability(NF, Delta_F, sigma, r, F_grid, K) 
