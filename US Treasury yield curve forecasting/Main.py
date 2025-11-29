################################################################################
################################################################################
#               This first section will only include PCA analysis components
#                   It corresponds to section 1
################################################################################
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

# --- 1. Load and clean data ---
file_path = "C:\\Users\\zelya\\OneDrive - Concordia University - Canada\\Documents\\Concordia\\Graduate Studies\\Courses\\STAT 380\\Project\\US Treasury Rates.xlsx"
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]

# Exclude 2020 and 2021
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]

# Drop missing values
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15  

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

# Split data
train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

# --- 4. Convert maturities to numerical values (in years) ---
maturity_map = {
    '6 Mo': 0.5,
    '1 Yr': 1,
    '2 Yr': 2,
    '3 Yr': 3,
    '5 Yr': 5,
    '7 Yr': 7,
    '10 Yr': 10,
    '20 Yr': 20,
    '30 Yr': 30
}
maturities = [maturity_map[col] for col in maturity_columns]
tau = np.array(maturities)  # maturities as array for x-axis

# --- 5. Prepare yield matrices and date arrays ---
Y_train = train_df[maturity_columns].to_numpy()
Y_val = val_df[maturity_columns].to_numpy()
Y_test = test_df[maturity_columns].to_numpy()

dates_train = train_df['Date'].to_numpy()
dates_val = val_df['Date'].to_numpy()
dates_test = test_df['Date'].to_numpy()

import matplotlib.pyplot as plt

for col in maturity_columns:
    plt.figure(figsize=(10, 6))
    plt.plot(df_filtered['Date'], df_filtered[col], label=col, color='tab:blue')
    plt.title(f"{col} Treasury Yield Over Time")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

#--------------- EXPLORATORY DATA ANALYSIS -------------------------------------
import seaborn as sns

# Choosing a representative subset of maturities
subset_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']

# Create scatterplots (pairplot)
sns.pairplot(df_filtered[subset_columns], height=2.5, aspect=1)
plt.suptitle("Scatterplot Matrix of Selected Treasury Maturities", y=1.02)
plt.tight_layout()
plt.show()


# Compute correlation matrix
corr_matrix = df_filtered[maturity_columns].corr()

# Plot heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.1)
plt.title("Correlation Heatmap of Treasury Yields")
plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------
#------------------------------- PCA ANALYSIS ----------------------------------
#-------------------------------------------------------------------------------

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# Step 1: Center the training data (PCA assumes zero-mean data)
Y_train_mean = Y_train.mean(axis=0)
Y_train_centered = Y_train - Y_train_mean

# Step 2: Fit PCA
pca = PCA()
pca.fit(Y_train_centered)

# Step 3: Determine # components to explain at least 99% of variance
#IMPORTANT I HAVE CHANGED 99% percent at the end of our analysis to capture more that is why the name is n_component_90 
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.99) + 1

# Components contributing to 99% variance
print(f"Number of components explaining 99% variance: {n_components_90}")

# Contributions of each maturity to the top components
contributions = np.abs(pca.components_[:n_components_90, :]).sum(axis=0)
contribution_percent = 100 * contributions / contributions.sum()

# Display top contributing maturities
maturity_contributions = pd.Series(contribution_percent, index=df_filtered.columns[1:])
maturity_contributions.sort_values(ascending=False).plot(kind='bar', figsize=(12, 5), title='Maturity Contributions to Top Principal Components (99% Variance)')
plt.ylabel('Contribution (%)')
plt.tight_layout()
plt.show()


# --- PCA Component Summary Table ---

# Calculate explained variance and cumulative variance
eigenvalues = pca.explained_variance_
variance_explained = pca.explained_variance_ratio_
cumulative_explained = np.cumsum(variance_explained)

# Create DataFrame for display
pca_summary_df = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(len(eigenvalues))],
    'Eigenvalue': np.round(eigenvalues, 4),
    'Variance Explained (%)': np.round(variance_explained * 100, 2),
    'Cumulative (%)': np.round(cumulative_explained * 100, 2)
})

# Display top N components (e.g., top 10)
top_n = 10
print(pca_summary_df.head(top_n))


# Step 4: Function to reconstruct yields using top components
def reconstruct_with_pca(Y, pca, n_components_90, mean_vector):
    Y_centered = Y - mean_vector
    scores = pca.transform(Y_centered)[:, :n_components_90]
    components = pca.components_[:n_components_90, :]
    Y_reconstructed = np.dot(scores, components) + mean_vector
    return Y_reconstructed

# Step 5: Reconstruct validation and test yields
Y_val_reconstructed = reconstruct_with_pca(Y_val, pca, n_components_90, Y_train_mean)
Y_test_reconstructed = reconstruct_with_pca(Y_test, pca, n_components_90, Y_train_mean)

# Step 6: Compute RMSE
rmse_val_pca = np.sqrt(mean_squared_error(Y_val, Y_val_reconstructed))
rmse_test_pca = np.sqrt(mean_squared_error(Y_test, Y_test_reconstructed))

print(f"Validation RMSE: {rmse_val_pca:.4f}")
print(f"Test RMSE: {rmse_test_pca:.4f}")

maturity_labels = maturity_columns  
dates_val = val_df['Date'].to_numpy()
dates_test = test_df['Date'].to_numpy()

# --- Plot actual vs reconstructed (validation set) ---
print("\nValidation Set - Actual vs Reconstructed Yields")

# Define sub-periods
val_dates_early = dates_val[dates_val <= np.datetime64('2019-12-31')]
val_dates_late = dates_val[dates_val >= np.datetime64('2022-01-01')]

# Indices for slicing
early_indices = np.where(dates_val <= np.datetime64('2019-12-31'))[0]
late_indices = np.where(dates_val >= np.datetime64('2022-01-01'))[0]

for i, label in enumerate(maturity_labels):
    # Early period
    plt.figure(figsize=(10, 4))
    plt.plot(val_dates_early, Y_val[early_indices, i], label='Actual', color='black')
    plt.plot(val_dates_early, Y_val_reconstructed[early_indices, i], label='Reconstructed (PCA)', linestyle='--', color='tab:blue')
    plt.title(f'Validation Set (Until 2019) - {label} Yield')
    plt.xlabel('Date')
    plt.ylabel('Yield (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Late period
    plt.figure(figsize=(10, 4))
    plt.plot(val_dates_late, Y_val[late_indices, i], label='Actual', color='black')
    plt.plot(val_dates_late, Y_val_reconstructed[late_indices, i], label='Reconstructed (PCA)', linestyle='--', color='tab:orange')
    plt.title(f'Validation Set (From 2022) - {label} Yield')
    plt.xlabel('Date')
    plt.ylabel('Yield (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --- Plot actual vs reconstructed (test set) ---
print("\n Test Set - Actual vs Reconstructed Yields")
for i, label in enumerate(maturity_labels):
    plt.figure(figsize=(10, 4))
    plt.plot(dates_test, Y_test[:, i], label='Actual', color='black')
    plt.plot(dates_test, Y_test_reconstructed[:, i], label='Reconstructed (PCA)', linestyle='--', color='tab:red')
    plt.title(f'Test Set - {label} Yield')
    plt.xlabel('Date')
    plt.ylabel('Yield (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
#-------------------------------------------------------------------------------
#-------------------------- Linear Model ---------------------------------------
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- STEP 1: Train on PCA scores ---
Z_train = pca.transform(Y_train_centered)[:, :n_components_90]
X_train_lag = Z_train[:-1]
Y_train_lead = Z_train[1:]

linear_model = LinearRegression()
linear_model.fit(X_train_lag, Y_train_lead)

print("\nLinear Model Coefficients (PCA space):")
print(linear_model.coef_)

# --- Predict on Train ---
Z_train_pred = linear_model.predict(X_train_lag)
Y_train_pred_centered = np.dot(Z_train_pred, pca.components_[:n_components_90, :])
Y_train_pred = Y_train_pred_centered + Y_train_mean
Y_train_actual = Y_train[1:]
dates_train_trimmed = dates_train[1:]

# --- Predict on Validation ---
Y_val_centered = Y_val - Y_train_mean
Z_val = pca.transform(Y_val_centered)[:, :n_components_90]
Z_val_lag = Z_val[:-1]
Z_val_pred = linear_model.predict(Z_val_lag)
Y_val_pred_centered = np.dot(Z_val_pred, pca.components_[:n_components_90, :])
Y_val_pred = Y_val_pred_centered + Y_train_mean
Y_val_actual = Y_val[1:]
dates_val_trimmed = dates_val[1:]

# --- Predict on Test ---
Y_test_centered = Y_test - Y_train_mean
Z_test = pca.transform(Y_test_centered)[:, :n_components_90]
Z_test_lag = Z_test[:-1]
Z_test_pred = linear_model.predict(Z_test_lag)
Y_test_pred_centered = np.dot(Z_test_pred, pca.components_[:n_components_90, :])
Y_test_pred = Y_test_pred_centered + Y_train_mean
Y_test_actual = Y_test[1:]
dates_test_trimmed = dates_test[1:]

# --- Compute RMSE ---
def compute_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

rmse_train_lm = compute_rmse(Y_train_actual, Y_train_pred)
rmse_val_lm = compute_rmse(Y_val_actual, Y_val_pred)
rmse_test_lm = compute_rmse(Y_test_actual, Y_test_pred)

print("\n----- RMSE (Root Mean Squared Error) -----")
print(f"Training RMSE:   {rmse_train_lm:.4f}")
print(f"Validation RMSE: {rmse_val_lm:.4f}")
print(f"Test RMSE:       {rmse_test_lm:.4f}")


# ------------------ Plot Training -------------------
for i in range(len(maturity_columns)):
    plt.figure(figsize=(10, 4))
    plt.plot(dates_train_trimmed, Y_train_actual[:, i], label="Actual", color='black')
    plt.plot(dates_train_trimmed, Y_train_pred[:, i], label="Predicted", linestyle='--', color='tab:orange')
    plt.title(f"Training Set – Maturity {i+1}")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#------------------ Validation plot ---------------------------------------
#Define Helper
def filter_by_year_range(dates, actual, predicted, start_year, end_year):
    years = pd.to_datetime(dates).year
    indices = np.where((years >= start_year) & (years <= end_year))[0]
    return dates[indices], actual[indices], predicted[indices]

# ------------------ Extract Validation Year Range ------------------
val_years = pd.to_datetime(dates_val_trimmed).year
val_start_year = val_years.min()
val_end_year = val_years.max()

# ------------------ Plot Validation ------------------
print("\nValidation Set – Actual vs Predicted Yields")

for i in range(len(maturity_columns)):
    # Plot 1: Start of validation to end of 2019
    dates_v1, Y_val_actual_v1, Y_val_pred_v1 = filter_by_year_range(
        dates_val_trimmed, Y_val_actual, Y_val_pred,
        start_year=val_start_year, end_year=2019
    )
    if len(dates_v1) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(dates_v1, Y_val_actual_v1[:, i], label="Actual", color='black')
        plt.plot(dates_v1, Y_val_pred_v1[:, i], label="Predicted", linestyle='--', color='tab:blue')
        plt.title(f"Validation Set (Start–2019) – Maturity {i+1}")
        plt.xlabel("Date")
        plt.ylabel("Yield (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    # Plot 2: 2022 to end of validation
    dates_v2, Y_val_actual_v2, Y_val_pred_v2 = filter_by_year_range(
        dates_val_trimmed, Y_val_actual, Y_val_pred,
        start_year=2022, end_year=val_end_year
    )
    if len(dates_v2) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(dates_v2, Y_val_actual_v2[:, i], label="Actual", color='black')
        plt.plot(dates_v2, Y_val_pred_v2[:, i], label="Predicted", linestyle='--', color='tab:red')
        plt.title(f"Validation Set (2022–End) – Maturity {i+1}")
        plt.xlabel("Date")
        plt.ylabel("Yield (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



# ------------------ Plot Test -------------------
for i in range(len(maturity_columns)):
    plt.figure(figsize=(10, 4))
    plt.plot(dates_test_trimmed, Y_test_actual[:, i], label="Actual", color='black')
    plt.plot(dates_test_trimmed, Y_test_pred[:, i], label="Predicted", linestyle='--', color='tab:green')
    plt.title(f"Test Set – Maturity {i+1}")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------
#--------------------------         LASSO        -------------------------------
#-------------------------------------------------------------------------------

# IMPORTANT!!!! : The lasso component was added here just to illustrate that it is redundent to use this method  due to the reasons discussed in the report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# -------------------- PCA Score Generation -----------------------
Z_train = pca.transform(Y_train_centered)[:, :n_components_90]
scores_train = Z_train  # PCA scores

# -------------------- Rolling Window CV to Find Best Alpha -------------------
alphas = np.logspace(-4, 1, 30)
tscv = TimeSeriesSplit(n_splits=5)
best_alpha = None
lowest_val_error = np.inf

for alpha in alphas:
    val_errors = []
    for train_idx, val_idx in tscv.split(scores_train[:-1]):
        X_train_cv, X_val_cv = scores_train[:-1][train_idx], scores_train[:-1][val_idx]
        Y_train_cv, Y_val_cv = scores_train[1:][train_idx], scores_train[1:][val_idx]

        Y_pred_cv = np.zeros_like(Y_val_cv)
        for m in range(Y_train_cv.shape[1]):
            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(X_train_cv, Y_train_cv[:, m])
            Y_pred_cv[:, m] = model.predict(X_val_cv)

        rmse = np.sqrt(mean_squared_error(Y_val_cv, Y_pred_cv))
        val_errors.append(rmse)

    avg_val_rmse = np.mean(val_errors)
    if avg_val_rmse < lowest_val_error:
        lowest_val_error = avg_val_rmse
        best_alpha = alpha

print("\n----- Best Alpha from Rolling CV -----")
print(f"Best alpha: {best_alpha:.5f} (Avg RMSE = {lowest_val_error:.4f})")

# -------------------- Train Final LASSO PCA Model -----------------------------
X_train_lag = scores_train[:-1]
Y_train_lead = scores_train[1:]
model_lasso = [Lasso(alpha=best_alpha, max_iter=10000).fit(X_train_lag, Y_train_lead[:, i]) for i in range(Y_train_lead.shape[1])]

# -------------------- Forecast PCA Scores (Train) -----------------------------
Z_train_pred = np.column_stack([model.predict(X_train_lag) for model in model_lasso])
Y_train_pred_centered = np.dot(Z_train_pred, pca.components_[:n_components_90, :])
Y_train_pred = Y_train_pred_centered + Y_train_mean
Y_train_actual = Y_train[1:]
dates_train_trimmed = dates_train[1:]

# -------------------- Validation Predictions -----------------------------
Y_val_centered = Y_val - Y_train_mean
Z_val = pca.transform(Y_val_centered)[:, :n_components_90]
Z_val_lag = Z_val[:-1]
Z_val_pred = np.column_stack([model.predict(Z_val_lag) for model in model_lasso])
Y_val_pred_centered = np.dot(Z_val_pred, pca.components_[:n_components_90, :])
Y_val_pred = Y_val_pred_centered + Y_train_mean
Y_val_actual = Y_val[1:]
dates_val_trimmed = dates_val[1:]

# -------------------- Test Predictions -----------------------------
Y_test_centered = Y_test - Y_train_mean
Z_test = pca.transform(Y_test_centered)[:, :n_components_90]
Z_test_lag = Z_test[:-1]
Z_test_pred = np.column_stack([model.predict(Z_test_lag) for model in model_lasso])
Y_test_pred_centered = np.dot(Z_test_pred, pca.components_[:n_components_90, :])
Y_test_pred = Y_test_pred_centered + Y_train_mean
Y_test_actual = Y_test[1:]
dates_test_trimmed = dates_test[1:]

# -------------------- RMSE -----------------------------
def compute_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

rmse_train_lml = compute_rmse(Y_train_actual, Y_train_pred)
rmse_val_lml = compute_rmse(Y_val_actual, Y_val_pred)
rmse_test_lml = compute_rmse(Y_test_actual, Y_test_pred)

print("\n----- RMSE (Root Mean Squared Error) -----")
print(f"Training RMSE:   {rmse_train_lml:.4f}")
print(f"Validation RMSE: {rmse_val_lml:.4f}")
print(f"Test RMSE:       {rmse_test_lml:.4f}")

# ------------------ Plot LASSO Training -------------------
print("\nLASSO – Training Set: Actual vs Predicted Yields")
maturities = Y_train.shape[1]  # Assuming Y_train has shape (n_samples, n_maturities)

for i in range(len(maturity_columns)):
    plt.figure(figsize=(10, 4))
    plt.plot(dates_train_trimmed, Y_train_actual[:, i], label="Actual", color='black')
    plt.plot(dates_train_trimmed, Y_train_pred[:, i], label="Predicted (LASSO)", linestyle='--', color='tab:orange')
    plt.title(f"LASSO – Training Set – Maturity {i+1}")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Validation plot (LASSO) ------------------
print("\nLASSO – Validation Set: Actual vs Predicted Yields")

def filter_by_year_range(dates, actual, predicted, start_year, end_year):
    years = pd.to_datetime(dates).year
    indices = np.where((years >= start_year) & (years <= end_year))[0]
    return dates[indices], actual[indices], predicted[indices]

val_years = pd.to_datetime(dates_val_trimmed).year
val_start_year = val_years.min()
val_end_year = val_years.max()

for i in range(len(maturity_columns)):
    # Plot 1: Start of validation to end of 2019
    dates_v1, Y_val_actual_v1, Y_val_pred_v1 = filter_by_year_range(
        dates_val_trimmed, Y_val_actual, Y_val_pred,
        start_year=val_start_year, end_year=2019
    )
    if len(dates_v1) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(dates_v1, Y_val_actual_v1[:, i], label="Actual", color='black')
        plt.plot(dates_v1, Y_val_pred_v1[:, i], label="Predicted (LASSO)", linestyle='--', color='tab:blue')
        plt.title(f"LASSO – Validation Set (Start–2019) – Maturity {i+1}")
        plt.xlabel("Date")
        plt.ylabel("Yield (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    # Plot 2: 2022 to end of validation
    dates_v2, Y_val_actual_v2, Y_val_pred_v2 = filter_by_year_range(
        dates_val_trimmed, Y_val_actual, Y_val_pred,
        start_year=2022, end_year=val_end_year
    )
    if len(dates_v2) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(dates_v2, Y_val_actual_v2[:, i], label="Actual", color='black')
        plt.plot(dates_v2, Y_val_pred_v2[:, i], label="Predicted (LASSO)", linestyle='--', color='tab:red')
        plt.title(f"LASSO – Validation Set (2022–End) – Maturity {i+1}")
        plt.xlabel("Date")
        plt.ylabel("Yield (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
# ------------------ Plot LASSO Test -------------------
print("\nLASSO – Test Set: Actual vs Predicted Yields")

for i in range(len(maturity_columns)):
    plt.figure(figsize=(10, 4))
    plt.plot(dates_test_trimmed, Y_test_actual[:, i], label="Actual", color='black')
    plt.plot(dates_test_trimmed, Y_test_pred[:, i], label="Predicted (LASSO)", linestyle='--', color='tab:green')
    plt.title(f"LASSO – Test Set – Maturity {i+1}")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
#-------------------------------------------------------------------------------
#------------------------  POLYNOMIAL REGRESSION (PCA) -------------------------
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# -------------------- PCA Score Generation -----------------------
Z_train = pca.transform(Y_train_centered)[:, :n_components_90]
scores_train = Z_train  # PCA scores

# -------------------- Rolling Window CV to Find Best Degree of Polynomial -------------------
degrees = [ 2, 3, 4, 5, 6, 7, 8, 9]  # Polynomial degrees to test
tscv = TimeSeriesSplit(n_splits=5)
best_degree = None
lowest_val_error = np.inf

for degree in degrees:
    poly = PolynomialFeatures(degree)
    scores_train_poly = poly.fit_transform(scores_train[:-1])  # Create polynomial features for PCA scores
    
    val_errors = []
    for train_idx, val_idx in tscv.split(scores_train_poly):
        X_train_cv, X_val_cv = scores_train_poly[train_idx], scores_train_poly[val_idx]
        Y_train_cv, Y_val_cv = scores_train[1:][train_idx], scores_train[1:][val_idx]

        Y_pred_cv = np.zeros_like(Y_val_cv)
        for m in range(Y_train_cv.shape[1]):
            model = LinearRegression()
            model.fit(X_train_cv, Y_train_cv[:, m])
            Y_pred_cv[:, m] = model.predict(X_val_cv)

        rmse = np.sqrt(mean_squared_error(Y_val_cv, Y_pred_cv))
        val_errors.append(rmse)

    avg_val_rmse = np.mean(val_errors)
    if avg_val_rmse < lowest_val_error:
        lowest_val_error = avg_val_rmse
        best_degree = degree

print("\n----- Best Polynomial Degree from Rolling CV -----")
print(f"Best degree: {best_degree} (Avg RMSE = {lowest_val_error:.4f})")

# -------------------- Train Final Polynomial Regression Model -----------------------------
poly = PolynomialFeatures(best_degree)
scores_train_poly = poly.fit_transform(scores_train[:-1])
X_train_lag = scores_train_poly
Y_train_lead = scores_train[1:]

model_poly = [LinearRegression().fit(X_train_lag, Y_train_lead[:, i]) for i in range(Y_train_lead.shape[1])]

# -------------------- Extract Features and Coefficients -----------------------------
# Feature names for polynomial terms
feature_names = poly.get_feature_names_out(input_features=[f"PC{i+1}" for i in range(scores_train.shape[1])])

# Extract coefficients for each polynomial regression model
coefficients = [model.coef_ for model in model_poly]

# Print out the features and their corresponding coefficients
for i, coef in enumerate(coefficients):
    print(f"\n----- Coefficients for Component {i+1} -----")
    feature_coeffs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef
    })
    feature_coeffs = feature_coeffs.sort_values(by='Coefficient', ascending=False)
    print(feature_coeffs)

# -------------------- Forecast PCA Scores (Train) -----------------------------
Z_train_pred = np.column_stack([model.predict(X_train_lag) for model in model_poly])
Y_train_pred_centered = np.dot(Z_train_pred, pca.components_[:n_components_90, :])
Y_train_pred = Y_train_pred_centered + Y_train_mean
Y_train_actual = Y_train[1:]
dates_train_trimmed = dates_train[1:]

# -------------------- Validation Predictions -----------------------------
Y_val_centered = Y_val - Y_train_mean
Z_val = pca.transform(Y_val_centered)[:, :n_components_90]
Z_val_poly = poly.transform(Z_val[:-1])
Z_val_pred = np.column_stack([model.predict(Z_val_poly) for model in model_poly])
Y_val_pred_centered = np.dot(Z_val_pred, pca.components_[:n_components_90, :])
Y_val_pred = Y_val_pred_centered + Y_train_mean
Y_val_actual = Y_val[1:]
dates_val_trimmed = dates_val[1:]

# -------------------- Test Predictions -----------------------------
Y_test_centered = Y_test - Y_train_mean
Z_test = pca.transform(Y_test_centered)[:, :n_components_90]
Z_test_poly = poly.transform(Z_test[:-1])
Z_test_pred = np.column_stack([model.predict(Z_test_poly) for model in model_poly])
Y_test_pred_centered = np.dot(Z_test_pred, pca.components_[:n_components_90, :])
Y_test_pred = Y_test_pred_centered + Y_train_mean
Y_test_actual = Y_test[1:]
dates_test_trimmed = dates_test[1:]

# -------------------- RMSE -----------------------------
def compute_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

rmse_train_pol = compute_rmse(Y_train_actual, Y_train_pred)
rmse_val_pol = compute_rmse(Y_val_actual, Y_val_pred)
rmse_test_pol = compute_rmse(Y_test_actual, Y_test_pred)

print("\n----- RMSE (Root Mean Squared Error) -----")
print(f"Training RMSE:   {rmse_train_pol:.4f}")
print(f"Validation RMSE: {rmse_val_pol:.4f}")
print(f"Test RMSE:       {rmse_test_pol:.4f}")

# -------------------- Error vs Degree Plot -----------------------------
val_errors_degrees = []

for degree in degrees:
    poly = PolynomialFeatures(degree)
    scores_train_poly = poly.fit_transform(scores_train[:-1])  # Create polynomial features for PCA scores
    
    val_errors = []
    for train_idx, val_idx in tscv.split(scores_train_poly):
        X_train_cv, X_val_cv = scores_train_poly[train_idx], scores_train_poly[val_idx]
        Y_train_cv, Y_val_cv = scores_train[1:][train_idx], scores_train[1:][val_idx]

        Y_pred_cv = np.zeros_like(Y_val_cv)
        for m in range(Y_train_cv.shape[1]):
            model = LinearRegression()
            model.fit(X_train_cv, Y_train_cv[:, m])
            Y_pred_cv[:, m] = model.predict(X_val_cv)

        rmse = np.sqrt(mean_squared_error(Y_val_cv, Y_pred_cv))
        val_errors.append(rmse)

    avg_val_rmse = np.mean(val_errors)
    val_errors_degrees.append(avg_val_rmse)

# Plot Degree vs RMSE (Error)
plt.figure(figsize=(8, 6))
plt.plot(degrees, val_errors_degrees, marker='o', linestyle='-', color='b')
plt.title('Polynomial Degree vs Validation RMSE (Error)')
plt.xlabel('Polynomial Degree')
plt.ylabel('Validation RMSE')
plt.grid(True)
plt.show()

# ------------------ Plot Polynomial Training -------------------
print("\nPolynomial Regression – Training Set: Actual vs Predicted Yields")
maturities = Y_train.shape[1]  # Assuming Y_train has shape (n_samples, n_maturities)

for i in range(maturities):
    plt.figure(figsize=(10, 4))
    plt.plot(dates_train_trimmed, Y_train_actual[:, i], label="Actual", color='black')
    plt.plot(dates_train_trimmed, Y_train_pred[:, i], label="Predicted (Polynomial)", linestyle='--', color='tab:orange')
    plt.title(f"Polynomial Regression – Training Set – Maturity {i+1}")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Validation Plot (Polynomial) ------------------
print("\nPolynomial Regression – Validation Set: Actual vs Predicted Yields")

for i in range(maturities):
    # Plot 1: Start of validation to end of 2019
    dates_v1, Y_val_actual_v1, Y_val_pred_v1 = filter_by_year_range(
        dates_val_trimmed, Y_val_actual, Y_val_pred,
        start_year=val_start_year, end_year=2019
    )
    if len(dates_v1) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(dates_v1, Y_val_actual_v1[:, i], label="Actual", color='black')
        plt.plot(dates_v1, Y_val_pred_v1[:, i], label="Predicted (Polynomial)", linestyle='--', color='tab:blue')
        plt.title(f"Polynomial Regression – Validation Set (Start–2019) – Maturity {i+1}")
        plt.xlabel("Date")
        plt.ylabel("Yield (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    # Plot 2: 2022 to end of validation
    dates_v2, Y_val_actual_v2, Y_val_pred_v2 = filter_by_year_range(
        dates_val_trimmed, Y_val_actual, Y_val_pred,
        start_year=2022, end_year=val_end_year
    )
    if len(dates_v2) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(dates_v2, Y_val_actual_v2[:, i], label="Actual", color='black')
        plt.plot(dates_v2, Y_val_pred_v2[:, i], label="Predicted (Polynomial)", linestyle='--', color='tab:red')
        plt.title(f"Polynomial Regression – Validation Set (2022–End) – Maturity {i+1}")
        plt.xlabel("Date")
        plt.ylabel("Yield (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ------------------ Plot Polynomial Test -------------------
print("\nPolynomial Regression – Test Set: Actual vs Predicted Yields")

for i in range(maturities):
    plt.figure(figsize=(10, 4))
    plt.plot(dates_test_trimmed, Y_test_actual[:, i], label="Actual", color='black')
    plt.plot(dates_test_trimmed, Y_test_pred[:, i], label="Predicted (Polynomial)", linestyle='--', color='tab:green')
    plt.title(f"Polynomial Regression – Test Set – Maturity {i+1}")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



################################################################################
#                     TRACKING DATAFRAME
################################################################################

# Compute RMSE for the Linear Model
rmse_train_lm = round(compute_rmse(Y_train_actual, Y_train_pred), 4)
rmse_val_lm = round(compute_rmse(Y_val_actual, Y_val_pred), 4)
rmse_test_lm = round(compute_rmse(Y_test_actual, Y_test_pred), 4)

# Compute RMSE for the LASSO Model on the Linear Model
rmse_train_lml = round(compute_rmse(Y_train_actual, Y_train_pred), 4)  # LASSO-based on linear model
rmse_val_lml = round(compute_rmse(Y_val_actual, Y_val_pred), 4)
rmse_test_lml = round(compute_rmse(Y_test_actual, Y_test_pred), 4)

# Compute RMSE for the Polynomial Model
rmse_train_pol = round(compute_rmse(Y_train_actual, Y_train_pred), 4)
rmse_val_pol = round(compute_rmse(Y_val_actual, Y_val_pred), 4)
rmse_test_pol = round(compute_rmse(Y_test_actual, Y_test_pred), 4)

# Now create the dictionary with all RMSE values
rmse_data = {
    'Model': ['Linear Model', 'Linear Model', 'Linear Model',
              'LASSO on Linear Model', 'LASSO on Linear Model', 'LASSO on Linear Model',
              'Polynomial Model', 'Polynomial Model', 'Polynomial Model'],
    'Set': ['Train', 'Validation', 'Test',
            'Train', 'Validation', 'Test',
            'Train', 'Validation', 'Test'],
    'RMSE': [rmse_train_lm, rmse_val_lm, rmse_test_lm,
             rmse_train_lml, rmse_val_lml, rmse_test_lml,
             rmse_train_pol, rmse_val_pol, rmse_test_pol]
}

# Convert the dictionary to a DataFrame
rmse_df = pd.DataFrame(rmse_data)

# Display the DataFrame
print(rmse_df)


################################################################################
################################################################################
#               This SECOND section will ONE-DAY LAG PREDICTION PROBLEM
#                   It corresponds to section 2 (1-day lag part only)
################################################################################
################################################################################

################################################################################
#--------------------- POLYNOMIAL REGRESSION -----------------------------------
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]

# Exclude 2020 and 2021
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]

# Drop missing values
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15  

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

# Split data
train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

# --- 4. Shift data for lagged values ---
train_yields = train_df[maturity_columns].values
val_yields = val_df[maturity_columns].values
test_yields = test_df[maturity_columns].values

train_lagged = train_yields[:-1]  # Lagged yields for training (shift by 1)
val_lagged = val_yields[:-1]  # Lagged yields for validation (shift by 1)
test_lagged = test_yields[:-1]  # Lagged yields for testing (shift by 1)

# Define the next time step as the target
train_target = train_yields[1:]  # Next yields for training
val_target = val_yields[1:]  # Next yields for validation
test_target = test_yields[1:]  # Next yields for testing


# --- 5. Test Polynomial Degrees and Evaluate RMSE ---
train_rmse, val_rmse, test_rmse = [], [], []
best_model = None
best_degree = None

for degree in range(1, 7):  # Test polynomial degrees from 1 to 6
    poly = PolynomialFeatures(degree=degree)
    
    # Apply polynomial transformation to the lagged data
    train_poly = poly.fit_transform(train_lagged)
    val_poly = poly.transform(val_lagged)
    test_poly = poly.transform(test_lagged)
    
    # Train model
    model = LinearRegression()
    model.fit(train_poly, train_target)
    
    # Predict on training, validation, and test data
    train_preds = model.predict(train_poly)
    val_preds = model.predict(val_poly)
    test_preds = model.predict(test_poly)
    
    # Compute RMSE for each maturity
    train_rmse.append(np.sqrt(mean_squared_error(train_target, train_preds)))
    val_rmse.append(np.sqrt(mean_squared_error(val_target, val_preds)))
    test_rmse.append(np.sqrt(mean_squared_error(test_target, test_preds)))
    
    # Check if this model is the best so far based on validation RMSE
    if best_degree is None or val_rmse[degree - 1] < val_rmse[best_degree - 1]:
        best_degree = degree
        best_model = model
        best_poly = poly  # Save the best polynomial transformer for later use

# --- 6. Identify Best Polynomial Degree Based on Validation RMSE ---
print(f"Best Polynomial Degree: {best_degree}")

# --- 7. Plot RMSE for each polynomial degree ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, 7), train_rmse, label='Train RMSE', marker='o')
plt.plot(range(1, 7), val_rmse, label='Validation RMSE', marker='o')
plt.plot(range(1, 7), test_rmse, label='Test RMSE', marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("RMSE for Polynomial Degrees (1-6) on Yield Curve Forecasting")
plt.legend()
plt.tight_layout()
plt.show()

# --- 8. Display RMSE Results ---
rmse_df = pd.DataFrame({
    'Polynomial Degree': range(1, 7),
    'Train RMSE': train_rmse,
    'Validation RMSE': val_rmse,
    'Test RMSE': test_rmse
})
print(rmse_df)

# --- 9. Extract and Display the Best Model's Coefficients ---
print("\nBest Model Coefficients:")
coefficients = best_model.coef_
intercept = best_model.intercept_

print(f"Intercept: {intercept}")
print(f"Coefficients for Polynomial Degree {best_degree}:")
print(coefficients)

# Maturity names (this should correspond to the number of rows in coefficients)
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']

# Polynomial degree (number of terms in your polynomial, based on the number of columns in coefficients)
poly_features = [f"Poly_{i}" for i in range(coefficients.shape[1])]

# Prepare a list to collect rows for the DataFrame
coef_list = []

# Loop through each maturity and its corresponding coefficients
for i, maturity in enumerate(maturity_columns):
    for j, poly_feature in enumerate(poly_features):
        coef_list.append({
            'Maturity': maturity,
            'Polynomial Feature': poly_feature,
            'Coefficient': coefficients[i][j]
        })

# Create the coefficients DataFrame
coef_df = pd.DataFrame(coef_list)

# Add the intercept for each maturity
intercept_df = pd.DataFrame({
    'Maturity': maturity_columns,
    'Intercept': intercept
})

# Merge the intercepts with the coefficients DataFrame
final_df = pd.merge(coef_df, intercept_df, on="Maturity", how="left")

# Display the final DataFrame
print(final_df)

#  Forecasting Next Day for All Maturities ---
def forecast_next_day(current_rates, model, poly_transformer):
    """
    Forecast the next day's rates based on current rates using the trained model and polynomial transformation.
    
    Parameters:
    current_rates - Array of current rates for all maturities
    model - Trained model (best model)
    poly_transformer - Polynomial transformer used in training
    
    Returns:
    Array of predicted rates for next day
    """
    # Apply the polynomial transformation to the current rates (reshape and transform)
    current_rates_reshaped = current_rates.reshape(1, -1)
    current_poly = poly_transformer.transform(current_rates_reshaped)
    
    # Forecast using the trained model
    next_day_rates = model.predict(current_poly)
    return next_day_rates[0]

# Forecast using the last available data point from the test set
last_observed_rates = test_df[maturity_columns].iloc[-1].values
next_day_forecast = forecast_next_day(last_observed_rates, best_model, best_poly)

# Create a DataFrame for forecast comparison
forecast_df = pd.DataFrame({
    'Maturity': maturity_columns,
    'Current Rate': last_observed_rates,
    'Forecasted Rate': next_day_forecast,
    'Change': next_day_forecast - last_observed_rates
})

# Print forecast for the next day
print("\nNext Day Forecast:")
print(forecast_df)

print(test_df.columns)

#-------------------------------------------------------------------------------
#                       RESULTS TRACKING
#-------------------------------------------------------------------------------

# RMSE values from our results
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Degree 1": [0.052817, 0.057270, 0.069784],
    "Degree 2": [0.051241, 0.093748, 0.156445],
    "Degree 3": [0.047875, 0.304437, 0.527379],
    "Degree 4": [0.038190, 6.570161, 7.997242],
    "Degree 5": [0.024071, 98.347837, 104.851405],
    "Degree 6": [0.019850, 172.631683, 346.306664],
})

# Round values for clarity
rmse_table = rmse_table.round(4)

# Show the table
print(rmse_table)



################################################################################
#--------------------- QUADRATIC SPLINE METHOD ---------------------------------
################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Rolling Window CV for best number of knots ---
# window_size defines the number of observations in each rolling window.
# stride controls how far forward the window moves after each iteration (non-overlapping segments).
# spline_degree is the degree of the spline basis functions.
# knot_range defines how many internal knots to test in the spline basis.

window_size = 750
stride = 250
spline_degree = 2
knot_range = range(5, 11)  # Try 5 to 10 knots

def create_lagged_data(yields): 
      # Create lagged predictor-response pairs.
      # X[t] predicts y[t+1], simulating a time series forecasting setup.
    X = yields[:-1]
    y = yields[1:]
    return X, y

rmse_per_knot = {k: [] for k in knot_range}

for start in range(0, len(df_filtered) - window_size - 1, stride):
  
    end = start + window_size
    window = df_filtered.iloc[start:end]
    X, y = create_lagged_data(window[maturity_columns].values)

    for n_knots in knot_range:
        val_errors = []
        for i in range(len(maturity_columns)):
            spline = SplineTransformer(degree=spline_degree, n_knots=n_knots, include_bias=False)
            X_spline = spline.fit_transform(X)
            coef, *_ = np.linalg.lstsq(X_spline, y[:, i], rcond=None)
            y_pred = X_spline @ coef
            val_errors.append(mean_squared_error(y[:, i], y_pred))
        avg_rmse = np.sqrt(np.mean(val_errors))
        rmse_per_knot[n_knots].append(avg_rmse)

# Average RMSE across all folds
avg_rmse_per_knot = {k: np.mean(v) for k, v in rmse_per_knot.items()}
best_k = min(avg_rmse_per_knot, key=avg_rmse_per_knot.get)
print(f"Best number of knots based on rolling CV: {best_k}")

# --- 4. Split final dataset ---
train_size = 0.7
val_size = 0.15
n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

train_X, train_y = create_lagged_data(train_df[maturity_columns].values)
val_X, val_y = create_lagged_data(val_df[maturity_columns].values)
test_X, test_y = create_lagged_data(test_df[maturity_columns].values)

# --- 5. Train on best number of knots ---
models = {}
train_preds, val_preds, test_preds = [], [], []
coefficients = {}

for i, maturity in enumerate(maturity_columns):
    spline = SplineTransformer(degree=spline_degree, n_knots=best_k, include_bias=False)
    X_train_spline = spline.fit_transform(train_X)
    coef, *_ = np.linalg.lstsq(X_train_spline, train_y[:, i], rcond=None)

    models[maturity] = {'spline': spline, 'coef': coef}
    coefficients[maturity] = coef

    train_preds.append(X_train_spline @ coef)
    val_preds.append(spline.transform(val_X) @ coef)
    test_preds.append(spline.transform(test_X) @ coef)

train_preds = np.column_stack(train_preds)
val_preds = np.column_stack(val_preds)
test_preds = np.column_stack(test_preds)

# --- 6. RMSE Summary ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

rmse_results = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [
        compute_rmse(train_y, train_preds),
        compute_rmse(val_y, val_preds),
        compute_rmse(test_y, test_preds)
    ]
})
print("\nMSE Summary:")
print(rmse_results)

# --- 7. Coefficients Summary ---
coef_df = pd.DataFrame(coefficients).T
coef_df.columns = [f'Spline_{i+1}' for i in range(coef_df.shape[1])]
coef_df.index.name = "Maturity"
print("\nCoefficients from Spline Models (OLS):")
print(coef_df.round(4))

#-------------------------------------------------------------------------------
#                     Smoothing constraint
#-------------------------------------------------------------------------------

# This section implements smoothing constraint splines by modifying the normal equations.
# The penalty matrix encourages smoothness by penalizing the second derivative of the spline.


# --- 1. Function to compute RMSE ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# --- 2. Function to fit smoothing spline and return RMSE on train/val/test along with coefficients ---
def fit_smoothing_spline(lambda_, n_knots, train_X, train_y, val_X, val_y, test_X, test_y, maturity_columns):
    train_preds, val_preds, test_preds = [], [], []
    coefficients = []

    for i, maturity in enumerate(maturity_columns):
        try:
            spline = SplineTransformer(degree=2, n_knots=n_knots, include_bias=False)
            X_train_spline = spline.fit_transform(train_X)
            X_val_spline = spline.transform(val_X)
            X_test_spline = spline.transform(test_X)
            
            # Compute second-difference penalty matrix.
            # Encourages adjacent spline coefficients to be similar (smoothness).

            n_basis = X_train_spline.shape[1]
            D = np.diff(np.eye(n_basis), n=2)
            P = D.T @ D

            XtX = X_train_spline.T @ X_train_spline
            Xty = X_train_spline.T @ train_y[:, i]

            # Align dimensions
            min_dim = min(P.shape[0], XtX.shape[0])
            P = P[:min_dim, :min_dim]
            XtX = XtX[:min_dim, :min_dim]
            Xty = Xty[:min_dim]

            beta = np.linalg.solve(XtX + lambda_ * P, Xty)

            train_preds.append(X_train_spline[:, :len(beta)] @ beta)
            val_preds.append(X_val_spline[:, :len(beta)] @ beta)
            test_preds.append(X_test_spline[:, :len(beta)] @ beta)
            coefficients.append(beta)  # Save the coefficients for this maturity

        except Exception as e:
            print(f"✗ Error at maturity {maturity}: {e}")
            return None, None  # Skip this config if it fails

    train_preds = np.column_stack(train_preds)
    val_preds = np.column_stack(val_preds)
    test_preds = np.column_stack(test_preds)

    # I will create a dataframe somewhat similar to the one before
    rmse_results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [compute_rmse(train_y, train_preds),
                 compute_rmse(val_y, val_preds),
                 compute_rmse(test_y, test_preds)]
    }

    # Convert RMSE results to DataFrame
    results_df = pd.DataFrame(rmse_results)

    # Convert coefficients to DataFrame
    coefficients_df = pd.DataFrame(coefficients).T  # Transpose to have one column per maturity
    coefficients_df.columns = [f"Coefficient_{i+1}" for i in range(coefficients_df.shape[1])]  # Rename columns

    return results_df, coefficients_df

# --- 3. Grid search with rolling window CV ---
def grid_search_smoothing(train_X, train_y, test_X, test_y, maturity_columns, lambda_range, knot_range, n_splits=5):
    best_config = None
    best_rmse = float('inf')
    best_rmse_results = None
    best_coefficients = None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_knots in knot_range:
        for lambda_ in lambda_range:
            print(f"Testing λ={lambda_:.2e}, knots={n_knots}")
            val_rmse_scores = []

            for train_index, val_index in tscv.split(train_X):
                X_train_cv, X_val_cv = train_X[train_index], train_X[val_index]
                y_train_cv, y_val_cv = train_y[train_index], train_y[val_index]

                rmse_results, coefficients = fit_smoothing_spline(lambda_, n_knots, X_train_cv, y_train_cv,
                                                                 X_val_cv, y_val_cv, test_X, test_y, maturity_columns)
                if rmse_results is None:
                    break  # Skip this config if any fold fails
                val_rmse_scores.append(rmse_results['RMSE'][1])  # Get the RMSE for validation as a scalar

            if len(val_rmse_scores) == n_splits:
                avg_val_rmse = np.mean(val_rmse_scores)
                if avg_val_rmse < best_rmse:
                    best_rmse = avg_val_rmse
                    best_config = (lambda_, n_knots)
                    best_rmse_results = rmse_results  # from last fold
                    best_coefficients = coefficients  # Store the coefficients for the best config

    print(f"\nTest config: λ={best_config[0]}, knots={best_config[1]}")
    print("\nRMSE for best config:")
    print(best_rmse_results)
    print("\nCoefficients for best config:")
    print(best_coefficients)
    return best_config, best_rmse_results, best_coefficients

lambda_range = np.logspace(-4, 4, 20)
knot_range = [3, 4, 5, 6, 7, 8, 9, 10]

best_config, best_rmse_results, best_coefficients = grid_search_smoothing(
    train_X, train_y, test_X, test_y,
    maturity_columns,
    lambda_range, knot_range,
    n_splits=5
)


#-------------------------------------------------------------------------------
#                             RIDGE ADDITION
#-------------------------------------------------------------------------------

# --- Grid of hyperparameters ---
lambda_values = np.logspace(-3, 3, 10)
alpha_values = np.logspace(-3, 3, 10)
n_knots_values = [3, 4, 5, 6, 7, 8, 9, 10]
spline_degree = 2
n_splits = 5  # number of rolling window folds

# --- Rolling window CV setup ---
tscv = TimeSeriesSplit(n_splits=n_splits)
best_rmse = float('inf')
best_params = {}

for n_knots in n_knots_values:
    for lambda_ in lambda_values:
        for alpha in alpha_values:
            rmse_vals = []

            for train_idx, val_idx in tscv.split(train_X):
                X_train_cv, X_val_cv = train_X[train_idx], train_X[val_idx]
                y_train_cv, y_val_cv = train_y[train_idx], train_y[val_idx]

                try:
                    # Build spline basis
                    spline = SplineTransformer(degree=spline_degree, n_knots=n_knots, include_bias=False)
                    X_train_spline = spline.fit_transform(X_train_cv)
                    X_val_spline = spline.transform(X_val_cv)

                    # Roughness penalty matrix
                    n_basis = X_train_spline.shape[1]
                    D = np.diff(np.eye(n_basis), n=2)
                    P = D.T @ D

                    # Align dimensions
                    if P.shape != (n_basis, n_basis):
                        min_dim = min(P.shape[0], n_basis)
                        P = P[:min_dim, :min_dim]
                        X_train_spline = X_train_spline[:, :min_dim]
                        X_val_spline = X_val_spline[:, :min_dim]

                    # Solve penalized least squares
                    XtX = X_train_spline.T @ X_train_spline
                    Xty = X_train_spline.T @ y_train_cv
                    I = np.eye(XtX.shape[0])
                    beta = np.linalg.solve(XtX + lambda_ * P + alpha * I, Xty)

                    # Predict and calculate RMSE
                    preds_val = X_val_spline @ beta
                    rmse_vals.append(np.sqrt(mean_squared_error(y_val_cv, preds_val)))

                except np.linalg.LinAlgError:
                    continue

            # Average RMSE across folds
            avg_rmse = np.mean(rmse_vals)

            # Keep best hyperparameters
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = {
                    'n_knots': n_knots,
                    'lambda_': lambda_,
                    'alpha': alpha
                }

# --- Output ---
print("Best Parameters:")
print(best_params)
print(f"Validation RMSE: {best_rmse:.4f}")

# I will now be running the model based on these results
# --- Configuration ---
spline_degree = 2
n_knots = 7
lambda_ = 0.001  # Smoothing parameter
alpha = 0.46415888336127775    # Ridge regularization parameter

models = {}
train_preds = []
val_preds = []
test_preds = []
coefficients = {}

for i, maturity in enumerate(maturity_columns):
    try:
        # --- Spline basis ---
        spline = SplineTransformer(degree=spline_degree, n_knots=n_knots, include_bias=False)
        X_spline = spline.fit_transform(train_X)

        # Construct roughness penalty matrix (2nd-order difference)
        n_basis = X_spline.shape[1]
        D = np.diff(np.eye(n_basis), n=2)
        P = D.T @ D  # Penalty matrix

        # Compute matrices
        XtX = X_spline.T @ X_spline
        Xty = X_spline.T @ train_y[:, i]

        # --- Align matrix shapes if needed ---
        if P.shape != XtX.shape:
            min_dim = min(P.shape[0], XtX.shape[0])
            P = P[:min_dim, :min_dim]
            XtX = XtX[:min_dim, :min_dim]
            Xty = Xty[:min_dim]

        # Solve penalized least squares: (X^T X + λP + αI)β = X^T y
        I = np.eye(XtX.shape[0])  # Identity matrix for Ridge
        beta = np.linalg.solve(XtX + lambda_ * P + alpha * I, Xty)

        # Store model
        models[maturity] = {'spline': spline, 'coef': beta}
        coefficients[maturity] = beta

        # Make predictions
        train_preds.append(spline.transform(train_X)[:, :len(beta)] @ beta)
        val_preds.append(spline.transform(val_X)[:, :len(beta)] @ beta)
        test_preds.append(spline.transform(test_X)[:, :len(beta)] @ beta)

        # Optional: Debug info
        print(f"✓ {maturity}: fitted with basis {n_basis}, beta shape {beta.shape}")

    except Exception as e:
        print(f"✗ Error at maturity {maturity}: {e}")

# --- Stack predictions ---
train_preds = np.column_stack(train_preds)
val_preds = np.column_stack(val_preds)
test_preds = np.column_stack(test_preds)

# --- RMSE Evaluation ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

rmse_results = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [
        compute_rmse(train_y, train_preds),
        compute_rmse(val_y, val_preds),
        compute_rmse(test_y, test_preds)
    ]
})

print("Smoothing Spline with Ridge RMSE Summary:")
print(rmse_results.round(6))

# --- Coefficients Summary ---
coef_df = pd.DataFrame(coefficients).T
coef_df.columns = [f'Spline_{i+1}' for i in range(coef_df.shape[1])]
coef_df.index.name = "Maturity"
print("\nSmoothing Spline with Ridge Coefficients:")
print(coef_df.round(4))

#-------------------------------------------------------------------------------
#                 TRACKING DATA FRAME
#-------------------------------------------------------------------------------

# RMSE values from our results
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Quadratic Spline (QS)": [0.066232, 0.140592, 1.899189],
    "Smoothed QS": [0.083491, 0.052320, 1.491030],
    "Smoothed Ridge QS": [0.098976, 0.152934, 1.019409]
})

# Round values for clarity
rmse_table = rmse_table.round(4)

# Show the table
print(rmse_table)



################################################################################
#                          CUBIC SPLINE METHOD  
################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Rolling Window CV for best number of knots ---
# window_size defines the number of observations in each rolling window.
# stride controls how far forward the window moves after each iteration (non-overlapping segments).
# spline_degree is the degree of the spline basis functions.
# knot_range defines how many internal knots to test in the spline basis.

window_size = 750
stride = 250
spline_degree = 3
knot_range = range(5, 11)  # Try 5 to 10 knots

def create_lagged_data(yields): 
      # Create lagged predictor-response pairs.
      # X[t] predicts y[t+1], simulating a time series forecasting setup.
    X = yields[:-1]
    y = yields[1:]
    return X, y

rmse_per_knot = {k: [] for k in knot_range}

for start in range(0, len(df_filtered) - window_size - 1, stride):
  
    end = start + window_size
    window = df_filtered.iloc[start:end]
    X, y = create_lagged_data(window[maturity_columns].values)

    for n_knots in knot_range:
        val_errors = []
        for i in range(len(maturity_columns)):
            spline = SplineTransformer(degree=spline_degree, n_knots=n_knots, include_bias=False)
            X_spline = spline.fit_transform(X)
            coef, *_ = np.linalg.lstsq(X_spline, y[:, i], rcond=None)
            y_pred = X_spline @ coef
            val_errors.append(mean_squared_error(y[:, i], y_pred))
        avg_rmse = np.sqrt(np.mean(val_errors))
        rmse_per_knot[n_knots].append(avg_rmse)

# Average RMSE across all folds
avg_rmse_per_knot = {k: np.mean(v) for k, v in rmse_per_knot.items()}
best_k = min(avg_rmse_per_knot, key=avg_rmse_per_knot.get)
print(f"Best number of knots based on rolling CV: {best_k}")

# --- 4. Split final dataset ---
train_size = 0.7
val_size = 0.15
n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

train_X, train_y = create_lagged_data(train_df[maturity_columns].values)
val_X, val_y = create_lagged_data(val_df[maturity_columns].values)
test_X, test_y = create_lagged_data(test_df[maturity_columns].values)

# --- 5. Train on best number of knots ---
models = {}
train_preds, val_preds, test_preds = [], [], []
coefficients = {}

for i, maturity in enumerate(maturity_columns):
    spline = SplineTransformer(degree=spline_degree, n_knots=best_k, include_bias=False)
    X_train_spline = spline.fit_transform(train_X)
    coef, *_ = np.linalg.lstsq(X_train_spline, train_y[:, i], rcond=None)

    models[maturity] = {'spline': spline, 'coef': coef}
    coefficients[maturity] = coef

    train_preds.append(X_train_spline @ coef)
    val_preds.append(spline.transform(val_X) @ coef)
    test_preds.append(spline.transform(test_X) @ coef)

train_preds = np.column_stack(train_preds)
val_preds = np.column_stack(val_preds)
test_preds = np.column_stack(test_preds)

# --- 6. RMSE Summary ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

rmse_results = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [
        compute_rmse(train_y, train_preds),
        compute_rmse(val_y, val_preds),
        compute_rmse(test_y, test_preds)
    ]
})
print("\nMSE Summary:")
print(rmse_results)

# --- 7. Coefficients Summary ---
coef_df = pd.DataFrame(coefficients).T
coef_df.columns = [f'Spline_{i+1}' for i in range(coef_df.shape[1])]
coef_df.index.name = "Maturity"
print("\nCoefficients from Spline Models (OLS):")
print(coef_df.round(4))

#-------------------------------------------------------------------------------
#                     Smoothing constraint
#-------------------------------------------------------------------------------

# This section implements smoothng constraint by modifying the normal equations.
# The penalty matrix encourages smoothness by penalizing the second derivative of the spline.


# --- 1. Function to compute RMSE ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# --- 2. Function to fit smoothing spline and return RMSE on train/val/test along with coefficients ---
def fit_smoothing_spline(lambda_, n_knots, train_X, train_y, val_X, val_y, test_X, test_y, maturity_columns):
    train_preds, val_preds, test_preds = [], [], []
    coefficients = []

    for i, maturity in enumerate(maturity_columns):
        try:
            spline = SplineTransformer(degree=3, n_knots=n_knots, include_bias=False)
            X_train_spline = spline.fit_transform(train_X)
            X_val_spline = spline.transform(val_X)
            X_test_spline = spline.transform(test_X)
            
            # Compute second-difference penalty matrix.
            # Encourages adjacent spline coefficients to be similar (smoothness).

            n_basis = X_train_spline.shape[1]
            D = np.diff(np.eye(n_basis), n=2)
            P = D.T @ D

            XtX = X_train_spline.T @ X_train_spline
            Xty = X_train_spline.T @ train_y[:, i]

            # Align dimensions
            min_dim = min(P.shape[0], XtX.shape[0])
            P = P[:min_dim, :min_dim]
            XtX = XtX[:min_dim, :min_dim]
            Xty = Xty[:min_dim]

            beta = np.linalg.solve(XtX + lambda_ * P, Xty)

            train_preds.append(X_train_spline[:, :len(beta)] @ beta)
            val_preds.append(X_val_spline[:, :len(beta)] @ beta)
            test_preds.append(X_test_spline[:, :len(beta)] @ beta)
            coefficients.append(beta)  # Save the coefficients for this maturity

        except Exception as e:
            print(f"✗ Error at maturity {maturity}: {e}")
            return None, None  # Skip this config if it fails

    train_preds = np.column_stack(train_preds)
    val_preds = np.column_stack(val_preds)
    test_preds = np.column_stack(test_preds)

    # I will create a dataframe somewhat similar to the one before
    rmse_results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [compute_rmse(train_y, train_preds),
                 compute_rmse(val_y, val_preds),
                 compute_rmse(test_y, test_preds)]
    }

    # Convert RMSE results to DataFrame
    results_df = pd.DataFrame(rmse_results)

    # Convert coefficients to DataFrame
    coefficients_df = pd.DataFrame(coefficients).T  # Transpose to have one column per maturity
    coefficients_df.columns = [f"Coefficient_{i+1}" for i in range(coefficients_df.shape[1])]  # Rename columns

    return results_df, coefficients_df

# --- 3. Grid search with rolling window CV ---
def grid_search_smoothing(train_X, train_y, test_X, test_y, maturity_columns, lambda_range, knot_range, n_splits=5):
    best_config = None
    best_rmse = float('inf')
    best_rmse_results = None
    best_coefficients = None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_knots in knot_range:
        for lambda_ in lambda_range:
            print(f"Testing λ={lambda_:.2e}, knots={n_knots}")
            val_rmse_scores = []

            for train_index, val_index in tscv.split(train_X):
                X_train_cv, X_val_cv = train_X[train_index], train_X[val_index]
                y_train_cv, y_val_cv = train_y[train_index], train_y[val_index]

                rmse_results, coefficients = fit_smoothing_spline(lambda_, n_knots, X_train_cv, y_train_cv,
                                                                 X_val_cv, y_val_cv, test_X, test_y, maturity_columns)
                if rmse_results is None:
                    break  # Skip this config if any fold fails
                val_rmse_scores.append(rmse_results['RMSE'][1])  # Get the RMSE for validation as a scalar

            if len(val_rmse_scores) == n_splits:
                avg_val_rmse = np.mean(val_rmse_scores)
                if avg_val_rmse < best_rmse:
                    best_rmse = avg_val_rmse
                    best_config = (lambda_, n_knots)
                    best_rmse_results = rmse_results  # from last fold
                    best_coefficients = coefficients  # Store the coefficients for the best config

    print(f"\nTest config: λ={best_config[0]}, knots={best_config[1]}")
    print("\nRMSE for best config:")
    print(best_rmse_results)
    print("\nCoefficients for best config:")
    print(best_coefficients)
    return best_config, best_rmse_results, best_coefficients

lambda_range = np.logspace(-4, 4, 20)
knot_range = [3, 4, 5, 6, 7, 8, 9, 10]

best_config, best_rmse_results, best_coefficients = grid_search_smoothing(
    train_X, train_y, test_X, test_y,
    maturity_columns,
    lambda_range, knot_range,
    n_splits=5
)


#-------------------------------------------------------------------------------
#                             RIDGE ADDITION
#-------------------------------------------------------------------------------
# --- Grid of hyperparameters ---
lambda_values = np.logspace(-3, 3, 10)
alpha_values = np.logspace(-3, 3, 10)
n_knots_values = [3, 4, 5, 6, 7, 8, 9, 10]
spline_degree = 3
n_splits = 5  # number of rolling window folds

# --- Rolling window CV setup ---
tscv = TimeSeriesSplit(n_splits=n_splits)
best_rmse = float('inf')
best_params = {}

for n_knots in n_knots_values:
    for lambda_ in lambda_values:
        for alpha in alpha_values:
            rmse_vals = []

            for train_idx, val_idx in tscv.split(train_X):
                X_train_cv, X_val_cv = train_X[train_idx], train_X[val_idx]
                y_train_cv, y_val_cv = train_y[train_idx], train_y[val_idx]

                try:
                    # Build spline basis
                    spline = SplineTransformer(degree=spline_degree, n_knots=n_knots, include_bias=False)
                    X_train_spline = spline.fit_transform(X_train_cv)
                    X_val_spline = spline.transform(X_val_cv)

                    # Roughness penalty matrix
                    n_basis = X_train_spline.shape[1]
                    D = np.diff(np.eye(n_basis), n=2)
                    P = D.T @ D

                    # Align dimensions
                    if P.shape != (n_basis, n_basis):
                        min_dim = min(P.shape[0], n_basis)
                        P = P[:min_dim, :min_dim]
                        X_train_spline = X_train_spline[:, :min_dim]
                        X_val_spline = X_val_spline[:, :min_dim]

                    # Solve penalized least squares
                    XtX = X_train_spline.T @ X_train_spline
                    Xty = X_train_spline.T @ y_train_cv
                    I = np.eye(XtX.shape[0])
                    beta = np.linalg.solve(XtX + lambda_ * P + alpha * I, Xty)

                    # Predict and calculate RMSE
                    preds_val = X_val_spline @ beta
                    rmse_vals.append(np.sqrt(mean_squared_error(y_val_cv, preds_val)))

                except np.linalg.LinAlgError:
                    continue

            # Average RMSE across folds
            avg_rmse = np.mean(rmse_vals)

            # Keep best hyperparameters
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_params = {
                    'n_knots': n_knots,
                    'lambda_': lambda_,
                    'alpha': alpha
                }

# --- Output ---
print("Best Parameters:")
print(best_params)
print(f"Validation RMSE: {best_rmse:.4f}")

# I will now be running the model based on these results
# --- Configuration ---
spline_degree = 3
n_knots = 3
lambda_ = 0.001  # Smoothing parameter
alpha = 0.021544346900318832    # Ridge regularization parameter

models = {}
train_preds = []
val_preds = []
test_preds = []
coefficients = {}

for i, maturity in enumerate(maturity_columns):
    try:
        # --- Spline basis ---
        spline = SplineTransformer(degree=spline_degree, n_knots=n_knots, include_bias=False)
        X_spline = spline.fit_transform(train_X)

        # Construct roughness penalty matrix (2nd-order difference)
        n_basis = X_spline.shape[1]
        D = np.diff(np.eye(n_basis), n=2)
        P = D.T @ D  # Penalty matrix

        # Compute matrices
        XtX = X_spline.T @ X_spline
        Xty = X_spline.T @ train_y[:, i]

        # --- Align matrix shapes if needed ---
        if P.shape != XtX.shape:
            min_dim = min(P.shape[0], XtX.shape[0])
            P = P[:min_dim, :min_dim]
            XtX = XtX[:min_dim, :min_dim]
            Xty = Xty[:min_dim]

        # Solve penalized least squares: (X^T X + λP + αI)β = X^T y
        I = np.eye(XtX.shape[0])  # Identity matrix for Ridge
        beta = np.linalg.solve(XtX + lambda_ * P + alpha * I, Xty)

        # Store model
        models[maturity] = {'spline': spline, 'coef': beta}
        coefficients[maturity] = beta

        # Make predictions
        train_preds.append(spline.transform(train_X)[:, :len(beta)] @ beta)
        val_preds.append(spline.transform(val_X)[:, :len(beta)] @ beta)
        test_preds.append(spline.transform(test_X)[:, :len(beta)] @ beta)

        # Optional: Debug info
        print(f"✓ {maturity}: fitted with basis {n_basis}, beta shape {beta.shape}")

    except Exception as e:
        print(f"✗ Error at maturity {maturity}: {e}")

# --- Stack predictions ---
train_preds = np.column_stack(train_preds)
val_preds = np.column_stack(val_preds)
test_preds = np.column_stack(test_preds)

# --- RMSE Evaluation ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

rmse_results = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [
        compute_rmse(train_y, train_preds),
        compute_rmse(val_y, val_preds),
        compute_rmse(test_y, test_preds)
    ]
})

print("Smoothing Spline with Ridge RMSE Summary:")
print(rmse_results.round(6))

# --- Coefficients Summary ---
coef_df = pd.DataFrame(coefficients).T
coef_df.columns = [f'Spline_{i+1}' for i in range(coef_df.shape[1])]
coef_df.index.name = "Maturity"
print("\nSmoothing Spline with Ridge Coefficients:")
print(coef_df.round(4))


#-------------------------------------------------------------------------------
#                 TRACKING DATA FRAME
#-------------------------------------------------------------------------------

# RMSE values from our results
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Cubic Spline (CS)": [0.051130, 0.190432, 0.540644],
    "Smoothed CS": [0.062726, 0.075273, 0.247285],
    "Smoothed Ridge CS": [0.062353, 0.086603, 0.192267]
})

# Round values for clarity
rmse_table = rmse_table.round(4)

# Show the table
print(rmse_table)


################################################################################
#---------------------------NEURAL NETWORKS------------------------------------ 
################################################################################


#-------------------------------------------------------------------------------
#               STACKED LSTM WTH INCREASING DROPOUTS
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

#setting the random seeds for reproducability, very important for the drop out rates!
np.random.seed(43)
torch.manual_seed(43)

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15

#now the following id for split indices
n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

#splitting the data accordingly
train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

#creating (x,y) paris so that input is today's yields and target is tmw's
def create_lagged(yield_data):
    return yield_data[:-1], yield_data[1:]

train_X, train_y = create_lagged(train_df[maturity_columns].values)
val_X, val_y = create_lagged(val_df[maturity_columns].values)
test_X, test_y = create_lagged(test_df[maturity_columns].values)

#helper function to convert numpy arrays to PyTorch tensors
def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

#direct application of that
train_X_tensor = to_tensor(train_X).unsqueeze(1)
train_y_tensor = to_tensor(train_y)
val_X_tensor = to_tensor(val_X).unsqueeze(1)
val_y_tensor = to_tensor(val_y)
test_X_tensor = to_tensor(test_X).unsqueeze(1)
test_y_tensor = to_tensor(test_y)

#creating data loaders for batching
#32 was a choice to balance memory usage and computational efficiency
batch_size = 32
train_loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size)

class YieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates):
        super(YieldLSTM, self).__init__()
        #the following is our architecture which is an alternatoin between LSTM and dropouts, ending with a fully connected layer to map to output
        #this is a three time stacked lstm, it allows to capture more complex temporal dependencies
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        self.fc = nn.Linear(hidden_sizes[2], input_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        #taking last time step output
        x = x[:, -1, :]
        return self.fc(x)

#instantiate theh model
input_size = len(maturity_columns)
hidden_sizes = [64, 128, 256] #number of hidden units for each lstm layer, a hierarchical approach used, as we increase complexity with each layer (can help better capturing long range dependencies and complex relationships in the time series data)
dropout_rates = [0.1, 0.2, 0.3] # increased number of dropout to match the increased complexity

model = YieldLSTM(input_size, hidden_sizes, dropout_rates)
criterion = nn.MSELoss()
#adam optimzer considered with a constant learning rate, adam adapts learning rates based on first and second moments, to stabilize training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#i thought the number 100 would give enough time for gradient to stabilize
num_epochs = 100
patience = 5 #patience is a parameter for early stopping ( no improvement after 5 epoch -> early stopping)
best_val_rmse = float('inf')
epochs_no_improve = 0
train_rmse_history = []
val_rmse_history = []
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []
# i wanted to now do the early stopping to stop when gradient seem to stop getting better
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_rmse = np.sqrt(np.mean(train_losses))
    train_rmse_history.append(train_rmse)

    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor)
        val_loss = criterion(val_preds, val_y_tensor)
        val_rmse = torch.sqrt(val_loss).item()

    val_rmse_history.append(val_rmse)

    print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-5:  # small tolerance to avoid float equality
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()
        epochs_no_improve = 0
        best_epoch = epoch
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

model.load_state_dict(best_model_state)

model.eval()
with torch.no_grad():
    train_preds = model(train_X_tensor)
    val_preds = model(val_X_tensor)
    test_preds = model(test_X_tensor)

train_rmse = np.sqrt(mean_squared_error(train_y, train_preds.numpy()))
val_rmse = np.sqrt(mean_squared_error(val_y, val_preds.numpy()))
test_rmse = np.sqrt(mean_squared_error(test_y, test_preds.numpy()))

train_mae = mean_absolute_error(train_y, train_preds.numpy())
val_mae = mean_absolute_error(val_y, val_preds.numpy())
test_mae = mean_absolute_error(test_y, test_preds.numpy())

results_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "RMSE": [train_rmse, val_rmse, test_rmse],
    "MAE": [train_mae, val_mae, test_mae]
})

print("\n--- RMSE and MAE Results ---")
print(results_df)

#to visualize our early stopping
plt.figure(figsize=(8, 5))
plt.plot(val_rmse_history, marker='o', label='Validation RMSE', color='blue')
plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch (Early Stopping)')
plt.title('Validation RMSE per Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#model parameters
print("\nModel Parameters by Layer:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} | Shape: {param.shape}")
        print(param.data)
        print("-" * 60)

##-------------------------------------------------------------------------------
#                           With Standarization
##-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

np.random.seed(43)
torch.manual_seed(43)

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

def create_lagged(yield_data):
    return yield_data[:-1], yield_data[1:]

train_X, train_y = create_lagged(train_df[maturity_columns].values)
val_X, val_y = create_lagged(val_df[maturity_columns].values)
test_X, test_y = create_lagged(test_df[maturity_columns].values)

# --- Standardize based on training set ---
train_mean_X = train_X.mean(axis=0)
train_std_X = train_X.std(axis=0)
train_mean_y = train_y.mean(axis=0)
train_std_y = train_y.std(axis=0)

train_std_X[train_std_X == 0] = 1e-6  #safesty measure to avoid dividing by zero
train_std_y[train_std_y == 0] = 1e-6  #safesty measure to avoid dividing by zero

train_X_std = (train_X - train_mean_X) / train_std_X
train_y_std = (train_y - train_mean_y) / train_std_y
val_X_std = (val_X - train_mean_X) / train_std_X
val_y_std = (val_y - train_mean_y) / train_std_y
test_X_std = (test_X - train_mean_X) / train_std_X
test_y_std = (test_y - train_mean_y) / train_std_y

def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

train_X_tensor = to_tensor(train_X_std).unsqueeze(1)
train_y_tensor = to_tensor(train_y_std)
val_X_tensor = to_tensor(val_X_std).unsqueeze(1)
val_y_tensor = to_tensor(val_y_std)
test_X_tensor = to_tensor(test_X_std).unsqueeze(1)
test_y_tensor = to_tensor(test_y_std)

batch_size = 32
train_loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size)

class YieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates):
        super(YieldLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        self.fc = nn.Linear(hidden_sizes[2], input_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        x = x[:, -1, :]
        return self.fc(x)

input_size = len(maturity_columns)
hidden_sizes = [64, 128, 256]
dropout_rates = [0.1, 0.2, 0.3]

model = YieldLSTM(input_size, hidden_sizes, dropout_rates)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
patience = 5
best_val_rmse = float('inf')
epochs_no_improve = 0
train_rmse_history = []
val_rmse_history = []
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_rmse = np.sqrt(np.mean(train_losses))
    train_rmse_history.append(train_rmse)

    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor)
        val_loss = criterion(val_preds, val_y_tensor)
        val_rmse = torch.sqrt(val_loss).item()

    val_rmse_history.append(val_rmse)

    print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-5:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()
        epochs_no_improve = 0
        best_epoch = epoch
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

model.load_state_dict(best_model_state)

# --- Inverse transform predictions before calculating metrics ---
model.eval()
with torch.no_grad():
    train_preds_std = model(train_X_tensor).numpy()
    val_preds_std = model(val_X_tensor).numpy()
    test_preds_std = model(test_X_tensor).numpy()

# Inverse transform
train_preds = train_preds_std * train_std_y + train_mean_y
val_preds = val_preds_std * train_std_y + train_mean_y
test_preds = test_preds_std * train_std_y + train_mean_y

train_y_orig = train_y_std * train_std_y + train_mean_y
val_y_orig = val_y_std * train_std_y + train_mean_y
test_y_orig = test_y_std * train_std_y + train_mean_y

train_rmse = np.sqrt(mean_squared_error(train_y_orig, train_preds))
val_rmse = np.sqrt(mean_squared_error(val_y_orig, val_preds))
test_rmse = np.sqrt(mean_squared_error(test_y_orig, test_preds))

train_mae = mean_absolute_error(train_y_orig, train_preds)
val_mae = mean_absolute_error(val_y_orig, val_preds)
test_mae = mean_absolute_error(test_y_orig, test_preds)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "RMSE": [train_rmse, val_rmse, test_rmse],
    "MAE": [train_mae, val_mae, test_mae]
})

print("\n--- RMSE and MAE Results ---")
print(results_df)

plt.figure(figsize=(8, 5))
plt.plot(val_rmse_history, marker='o', label='Validation RMSE', color='blue')
plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch (Early Stopping)')
plt.title('Validation RMSE per Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nModel Parameters by Layer:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} | Shape: {param.shape}")
        print(param.data)
        print("-" * 60)


#-------------------------------------------------------------------------------
#                       Weights sharing
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

np.random.seed(43)
torch.manual_seed(43)

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

def create_lagged(yield_data):
    return yield_data[:-1], yield_data[1:]

train_X, train_y = create_lagged(train_df[maturity_columns].values)
val_X, val_y = create_lagged(val_df[maturity_columns].values)
test_X, test_y = create_lagged(test_df[maturity_columns].values)

# --- Standardize based on training set ---
train_mean_X = train_X.mean(axis=0)
train_std_X = train_X.std(axis=0)
train_mean_y = train_y.mean(axis=0)
train_std_y = train_y.std(axis=0)

train_std_X[train_std_X == 0] = 1e-6
train_std_y[train_std_y == 0] = 1e-6

train_X_std = (train_X - train_mean_X) / train_std_X
train_y_std = (train_y - train_mean_y) / train_std_y
val_X_std = (val_X - train_mean_X) / train_std_X
val_y_std = (val_y - train_mean_y) / train_std_y
test_X_std = (test_X - train_mean_X) / train_std_X
test_y_std = (test_y - train_mean_y) / train_std_y

def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

train_X_tensor = to_tensor(train_X_std).unsqueeze(1)
train_y_tensor = to_tensor(train_y_std)
val_X_tensor = to_tensor(val_X_std).unsqueeze(1)
val_y_tensor = to_tensor(val_y_std)
test_X_tensor = to_tensor(test_X_std).unsqueeze(1)
test_y_tensor = to_tensor(test_y_std)

batch_size = 32
train_loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size)

class YieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(YieldLSTM, self).__init__()
        # Shared LSTM weight
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Apply the shared LSTM weights for all layers
        x, _ = self.lstm(x)  # Pass through shared LSTM layer
        x = self.dropout(x)
        x = x[:, -1, :]  # Use the last hidden state
        return self.fc(x)

input_size = len(maturity_columns)
hidden_size = 128
dropout_rate = 0.2

model = YieldLSTM(input_size, hidden_size, dropout_rate)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
patience = 5
best_val_rmse = float('inf')
epochs_no_improve = 0
train_rmse_history = []
val_rmse_history = []
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_rmse = np.sqrt(np.mean(train_losses))
    train_rmse_history.append(train_rmse)

    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor)
        val_loss = criterion(val_preds, val_y_tensor)
        val_rmse = torch.sqrt(val_loss).item()

    val_rmse_history.append(val_rmse)

    print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-5:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()
        epochs_no_improve = 0
        best_epoch = epoch
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

model.load_state_dict(best_model_state)

# --- Inverse transform predictions before calculating metrics ---
model.eval()
with torch.no_grad():
    train_preds_std = model(train_X_tensor).numpy()
    val_preds_std = model(val_X_tensor).numpy()
    test_preds_std = model(test_X_tensor).numpy()

# Inverse transform
train_preds = train_preds_std * train_std_y + train_mean_y
val_preds = val_preds_std * train_std_y + train_mean_y
test_preds = test_preds_std * train_std_y + train_mean_y

train_y_orig = train_y_std * train_std_y + train_mean_y
val_y_orig = val_y_std * train_std_y + train_mean_y
test_y_orig = test_y_std * train_std_y + train_mean_y

train_rmse = np.sqrt(mean_squared_error(train_y_orig, train_preds))
val_rmse = np.sqrt(mean_squared_error(val_y_orig, val_preds))
test_rmse = np.sqrt(mean_squared_error(test_y_orig, test_preds))

train_mae = mean_absolute_error(train_y_orig, train_preds)
val_mae = mean_absolute_error(val_y_orig, val_preds)
test_mae = mean_absolute_error(test_y_orig, test_preds)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "RMSE": [train_rmse, val_rmse, test_rmse],
    "MAE": [train_mae, val_mae, test_mae]
})

print("\n--- RMSE and MAE Results ---")
print(results_df)

plt.figure(figsize=(8, 5))
plt.plot(val_rmse_history, marker='o', label='Validation RMSE', color='blue')
plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch (Early Stopping)')
plt.title('Validation RMSE per Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nModel Parameters by Layer:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} | Shape: {param.shape}")
        print(param.data)
        print("-" * 60)
        
        
#-------------------------------------------------------------------------------
#                           Result Tracking
#-------------------------------------------------------------------------------

# RMSE values from our results
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "LSTM + Dropouts (LD)": [0.114127, 0.276051, 0.391191],
    "Standarized LD (SLD)": [0.102592, 0.223080, 0.355892],
    "Weight sharing SLD": [0.063157, 0.118590, 0.116687]
})

# Round values for clarity
rmse_table = rmse_table.round(4)

print("\n RMSE Tracking table\n")
# Show the table
print(rmse_table)


# RMSE values from our results
mae_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "LSTM + Dropouts (LD)": [0.086504, 0.219761, 0.329148],
    "Standarized LD (SLD)": [0.074371, 0.167850, 0.239791],
    "Weight sharing SLD": [0.045019, 0.084770, 0.091996]
})

# Round values for clarity
mae_table = mae_table.round(4)

print("\n MAE Tracking table\n")
# Show the table
print(mae_table)

################################################################################
#---------------------------ALGORITHM COMPARISON-------------------------------- 
################################################################################
import pandas as pd
import matplotlib.pyplot as plt

# RMSE values from our results
#degrees 4 to 6 omitted due to the very large test rmse
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Degree 1": [0.052817, 0.057270, 0.069784],
    "Degree 2": [0.051241, 0.093748, 0.156445],
    "Degree 3": [0.047875, 0.304437, 0.527379],
    "QS": [0.066232, 0.140592, 1.899189],
    "SQS": [0.083491, 0.052320, 1.491030],
    "SRQS": [0.098976, 0.152934, 1.019409],
    "CS": [0.051130, 0.190432, 0.540644],
    "SCS": [0.062726, 0.075273, 0.247285],
    "SRCS": [0.062353, 0.086603, 0.192267],
    "LD": [0.114127, 0.276051, 0.391191],
    "SLD": [0.102592, 0.223080, 0.355892],
    "WSLD": [0.063157, 0.118590, 0.116687]
})


# Define the list of methods (column names to compare)
methods = rmse_table.columns[1:]  # Skip the first column 'Dataset'

# Extract the Test RMSE values for each method
test_rmse_values = [rmse_table.loc[rmse_table["Dataset"] == "Test", method].values[0] for method in methods]

# Create a bar chart
fig = plt.figure()
fig.suptitle('Algorithm Error Comparison , (Test RMSE, the lower the better)')
ax = fig.add_subplot(111)

# Plot the bar chart
bars = ax.bar(methods, test_rmse_values)

# Rotate the x-axis labels for better readability
ax.set_xticklabels(methods, rotation=90)

# Set axis labels
ax.set_xlabel('Methods')
ax.set_ylabel('Test RMSE')

# Annotate each bar with the RMSE value above it (rounded to 3 decimal places)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

# Adjust plot size and display
fig.set_size_inches(15,8)
plt.show()



################################################################################
################################################################################
#               This THIRD section will BE ON FIVE-DAY LAG PREDICTION PROBLEM
#                   It corresponds to section 2 (5-day lag part only)
################################################################################
################################################################################


################################################################################
#--------------------- POLYNOMIAL REGRESSION -----------------------------------
################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]

# Exclude 2020 and 2021
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]

# Drop missing values
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15  

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

# Split data
train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

# --- 4. Shift data for lagged values ---
lag = 5  # Define lag of 5 days

# Lagged yields for training, validation, and testing (shift by 5 days)
train_lagged = train_df[maturity_columns].shift(lag).dropna().values
val_lagged = val_df[maturity_columns].shift(lag).dropna().values
test_lagged = test_df[maturity_columns].shift(lag).dropna().values

# Define the next time step as the target (5 days after lag)
train_target = train_df[maturity_columns].iloc[lag:].values
val_target = val_df[maturity_columns].iloc[lag:].values
test_target = test_df[maturity_columns].iloc[lag:].values


# --- 5. Test Polynomial Degrees and Evaluate RMSE ---
train_rmse, val_rmse, test_rmse = [], [], []
best_model = None
best_degree = None

for degree in range(1, 7):  # Test polynomial degrees from 1 to 6
    poly = PolynomialFeatures(degree=degree)
    
    # Apply polynomial transformation to the lagged data
    train_poly = poly.fit_transform(train_lagged)
    val_poly = poly.transform(val_lagged)
    test_poly = poly.transform(test_lagged)
    
    # Train model
    model = LinearRegression()
    model.fit(train_poly, train_target)
    
    # Predict on training, validation, and test data
    train_preds = model.predict(train_poly)
    val_preds = model.predict(val_poly)
    test_preds = model.predict(test_poly)
    
    # Compute RMSE for each maturity
    train_rmse.append(np.sqrt(mean_squared_error(train_target, train_preds)))
    val_rmse.append(np.sqrt(mean_squared_error(val_target, val_preds)))
    test_rmse.append(np.sqrt(mean_squared_error(test_target, test_preds)))
    
    # Check if this model is the best so far based on validation RMSE
    if best_degree is None or val_rmse[degree - 1] < val_rmse[best_degree - 1]:
        best_degree = degree
        best_model = model
        best_poly = poly  # Save the best polynomial transformer for later use

# --- 6. Identify Best Polynomial Degree Based on Validation RMSE ---
print(f"Best Polynomial Degree: {best_degree}")

# --- 7. Plot RMSE for each polynomial degree ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, 7), train_rmse, label='Train RMSE', marker='o')
plt.plot(range(1, 7), val_rmse, label='Validation RMSE', marker='o')
plt.plot(range(1, 7), test_rmse, label='Test RMSE', marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("RMSE for Polynomial Degrees (1-6) on Yield Curve Forecasting")
plt.legend()
plt.tight_layout()
plt.show()

# --- 8. Display RMSE Results ---
rmse_df = pd.DataFrame({
    'Polynomial Degree': range(1, 7),
    'Train RMSE': train_rmse,
    'Validation RMSE': val_rmse,
    'Test RMSE': test_rmse
})
print(rmse_df)

# --- 9. Extract and Display the Best Model's Coefficients ---
print("\nBest Model Coefficients:")
coefficients = best_model.coef_
intercept = best_model.intercept_

print(f"Intercept: {intercept}")
print(f"Coefficients for Polynomial Degree {best_degree}:")
print(coefficients)

# Maturity names (this should correspond to the number of rows in coefficients)
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']

# Polynomial degree (number of terms in your polynomial, based on the number of columns in coefficients)
poly_features = [f"Poly_{i}" for i in range(coefficients.shape[1])]

# Prepare a list to collect rows for the DataFrame
coef_list = []

# Loop through each maturity and its corresponding coefficients
for i, maturity in enumerate(maturity_columns):
    for j, poly_feature in enumerate(poly_features):
        coef_list.append({
            'Maturity': maturity,
            'Polynomial Feature': poly_feature,
            'Coefficient': coefficients[i][j]
        })

# Create the coefficients DataFrame
coef_df = pd.DataFrame(coef_list)

# Add the intercept for each maturity
intercept_df = pd.DataFrame({
    'Maturity': maturity_columns,
    'Intercept': intercept
})

# Merge the intercepts with the coefficients DataFrame
final_df = pd.merge(coef_df, intercept_df, on="Maturity", how="left")

# Display the final DataFrame
print(final_df)

#  Forecasting Next Day for All Maturities ---
def forecast_next_day(current_rates, model, poly_transformer):
    """
    Forecast the next day's rates based on current rates using the trained model and polynomial transformation.
    
    Parameters:
    current_rates - Array of current rates for all maturities
    model - Trained model (best model)
    poly_transformer - Polynomial transformer used in training
    
    Returns:
    Array of predicted rates for next day
    """
    # Apply the polynomial transformation to the current rates (reshape and transform)
    current_rates_reshaped = current_rates.reshape(1, -1)
    current_poly = poly_transformer.transform(current_rates_reshaped)
    
    # Forecast using the trained model
    next_day_rates = model.predict(current_poly)
    return next_day_rates[0]

# Forecast using the last available data point from the test set
last_observed_rates = test_df[maturity_columns].iloc[-1].values
next_day_forecast = forecast_next_day(last_observed_rates, best_model, best_poly)

# Create a DataFrame for forecast comparison
forecast_df = pd.DataFrame({
    'Maturity': maturity_columns,
    'Current Rate': last_observed_rates,
    'Forecasted Rate': next_day_forecast,
    'Change': next_day_forecast - last_observed_rates
})

# Print forecast for the next day
print("\nNext Day Forecast:")
print(forecast_df)


################################################################################
#--------------------- QUADRATIC SPLINE METHODS---------------------------------
################################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Rolling Window CV for best number of knots ---
# window_size defines the number of observations in each rolling window.
# stride controls how far forward the window moves after each iteration (non-overlapping segments).
# spline_degree is the degree of the spline basis functions.
# knot_range defines how many internal knots to test in the spline basis.

window_size = 750
stride = 250
spline_degree = 2
knot_range = range(3, 11)  # Try 3 to 10 knots

def create_lagged_data(yields, lag=5): 
    """Create lagged predictor-response pairs with a specified lag (default 5 days)."""
    X = yields[:-lag]  # Shift the predictor by 'lag' days
    y = yields[lag:]   # Shift the target by 'lag' days
    return X, y

rmse_per_knot = {k: [] for k in knot_range}

for start in range(0, len(df_filtered) - window_size - 1, stride):
  
    end = start + window_size
    window = df_filtered.iloc[start:end]
    X, y = create_lagged_data(window[maturity_columns].values, lag=5)  # Pass the 5-day lag

    for n_knots in knot_range:
        val_errors = []
        for i in range(len(maturity_columns)):
            spline = SplineTransformer(degree=spline_degree, n_knots=n_knots, include_bias=False)
            X_spline = spline.fit_transform(X)
            coef, *_ = np.linalg.lstsq(X_spline, y[:, i], rcond=None)
            y_pred = X_spline @ coef
            val_errors.append(mean_squared_error(y[:, i], y_pred))
        avg_rmse = np.sqrt(np.mean(val_errors))
        rmse_per_knot[n_knots].append(avg_rmse)

# Average RMSE across all folds
avg_rmse_per_knot = {k: np.mean(v) for k, v in rmse_per_knot.items()}
best_k = min(avg_rmse_per_knot, key=avg_rmse_per_knot.get)
print(f"Best number of knots based on rolling CV: {best_k}")

# --- 4. Split final dataset ---
train_size = 0.7
val_size = 0.15
n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

train_X, train_y = create_lagged_data(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged_data(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged_data(test_df[maturity_columns].values, lag=5)

# --- 5. Train on best number of knots ---
models = {}
train_preds, val_preds, test_preds = [], [], []
coefficients = {}

for i, maturity in enumerate(maturity_columns):
    spline = SplineTransformer(degree=spline_degree, n_knots=best_k, include_bias=False)
    X_train_spline = spline.fit_transform(train_X)
    coef, *_ = np.linalg.lstsq(X_train_spline, train_y[:, i], rcond=None)

    models[maturity] = {'spline': spline, 'coef': coef}
    coefficients[maturity] = coef

    train_preds.append(X_train_spline @ coef)
    val_preds.append(spline.transform(val_X) @ coef)
    test_preds.append(spline.transform(test_X) @ coef)

train_preds = np.column_stack(train_preds)
val_preds = np.column_stack(val_preds)
test_preds = np.column_stack(test_preds)

# --- 6. RMSE Summary ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

rmse_results = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [
        compute_rmse(train_y, train_preds),
        compute_rmse(val_y, val_preds),
        compute_rmse(test_y, test_preds)
    ]
})
print("\nMSE Summary:")
print(rmse_results)

# --- 7. Coefficients Summary ---
coef_df = pd.DataFrame(coefficients).T
coef_df.columns = [f'Spline_{i+1}' for i in range(coef_df.shape[1])]
coef_df.index.name = "Maturity"
print("\nCoefficients from Spline Models (OLS):")
print(coef_df.round(4))

#-------------------------------------------------------------------------------
#                     Smoothing constraint
#-------------------------------------------------------------------------------

train_X, train_y = create_lagged_data(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged_data(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged_data(test_df[maturity_columns].values, lag=5)

# --- 1. Function to compute RMSE ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# --- 2. Function to fit smoothing spline and return RMSE on train/val/test along with coefficients ---
def fit_smoothing_spline(lambda_, n_knots, train_X, train_y, val_X, val_y, test_X, test_y, maturity_columns):
    train_preds, val_preds, test_preds = [], [], []
    coefficients = []

    for i, maturity in enumerate(maturity_columns):
        try:
            spline = SplineTransformer(degree=2, n_knots=n_knots, include_bias=False)
            X_train_spline = spline.fit_transform(train_X)
            X_val_spline = spline.transform(val_X)
            X_test_spline = spline.transform(test_X)
            
            # Compute second-difference penalty matrix.
            # Encourages adjacent spline coefficients to be similar (smoothness).

            n_basis = X_train_spline.shape[1]
            D = np.diff(np.eye(n_basis), n=2)
            P = D.T @ D

            XtX = X_train_spline.T @ X_train_spline
            Xty = X_train_spline.T @ train_y[:, i]

            # Align dimensions
            min_dim = min(P.shape[0], XtX.shape[0])
            P = P[:min_dim, :min_dim]
            XtX = XtX[:min_dim, :min_dim]
            Xty = Xty[:min_dim]

            beta = np.linalg.solve(XtX + lambda_ * P, Xty)

            train_preds.append(X_train_spline[:, :len(beta)] @ beta)
            val_preds.append(X_val_spline[:, :len(beta)] @ beta)
            test_preds.append(X_test_spline[:, :len(beta)] @ beta)
            coefficients.append(beta)  # Save the coefficients for this maturity

        except Exception as e:
            print(f"✗ Error at maturity {maturity}: {e}")
            return None, None  # Skip this config if it fails

    train_preds = np.column_stack(train_preds)
    val_preds = np.column_stack(val_preds)
    test_preds = np.column_stack(test_preds)

    # I will create a dataframe somewhat similar to the one before
    rmse_results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [compute_rmse(train_y, train_preds),
                 compute_rmse(val_y, val_preds),
                 compute_rmse(test_y, test_preds)]
    }

    # Convert RMSE results to DataFrame
    results_df = pd.DataFrame(rmse_results)

    # Convert coefficients to DataFrame
    coefficients_df = pd.DataFrame(coefficients).T  # Transpose to have one column per maturity
    coefficients_df.columns = [f"Coefficient_{i+1}" for i in range(coefficients_df.shape[1])]  # Rename columns

    return results_df, coefficients_df

# --- 3. Grid search with rolling window CV ---
def grid_search_smoothing(train_X, train_y, test_X, test_y, maturity_columns, lambda_range, knot_range, n_splits=5):
    best_config = None
    best_rmse = float('inf')
    best_rmse_results = None
    best_coefficients = None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_knots in knot_range:
        for lambda_ in lambda_range:
            print(f"Testing λ={lambda_:.2e}, knots={n_knots}")
            val_rmse_scores = []

            for train_index, val_index in tscv.split(train_X):
                X_train_cv, X_val_cv = train_X[train_index], train_X[val_index]
                y_train_cv, y_val_cv = train_y[train_index], train_y[val_index]

                rmse_results, coefficients = fit_smoothing_spline(lambda_, n_knots, X_train_cv, y_train_cv,
                                                                 X_val_cv, y_val_cv, test_X, test_y, maturity_columns)
                if rmse_results is None:
                    break  # Skip this config if any fold fails
                val_rmse_scores.append(rmse_results['RMSE'][1])  # Get the RMSE for validation as a scalar

            if len(val_rmse_scores) == n_splits:
                avg_val_rmse = np.mean(val_rmse_scores)
                if avg_val_rmse < best_rmse:
                    best_rmse = avg_val_rmse
                    best_config = (lambda_, n_knots)
                    best_rmse_results = rmse_results  # from last fold
                    best_coefficients = coefficients  # Store the coefficients for the best config

    print(f"\nTest config: λ={best_config[0]}, knots={best_config[1]}")
    print("\nRMSE for best config:")
    print(best_rmse_results)
    print("\nCoefficients for best config:")
    print(best_coefficients)
    return best_config, best_rmse_results, best_coefficients

lambda_range = np.logspace(-4, 4, 20)
knot_range = [3, 4, 5, 6, 7, 8, 9, 10]

best_config, best_rmse_results, best_coefficients = grid_search_smoothing(
    train_X, train_y, test_X, test_y,
    maturity_columns,
    lambda_range, knot_range,
    n_splits=5
)

#-------------------------------------------------------------------------------
#                             RIDGE + SMOOTHING
#-------------------------------------------------------------------------------
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

# Step 1: Create lagged data
def create_lagged_data(data, lag):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i])  # Previous 'lag' observations as features
        y.append(data[i])          # Current observation as target
    return np.array(X), np.array(y)

# Step 2: Compute RMSE
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# Step 3: Ridge + Smoothing function (with lambda, alpha, and knots)
def fit_ridge_smoothing(lambda_, alpha_, n_knots, train_X, train_y, val_X, val_y, test_X, test_y, maturity_columns):
    train_preds, val_preds, test_preds = [], [], []
    coefficients = []

    for i, maturity in enumerate(maturity_columns):
        try:
            # Apply smoothing spline transformation
            spline = SplineTransformer(degree=2, n_knots=n_knots, include_bias=False)
            X_train_spline = spline.fit_transform(train_X)
            X_val_spline = spline.transform(val_X)
            X_test_spline = spline.transform(test_X)
            
            # Precompute terms
            XtX = X_train_spline.T @ X_train_spline
            Xty = X_train_spline.T @ train_y[:, i]
            I = np.eye(XtX.shape[0])

            # Build smoothing penalty matrix P
            # P = D^T D where D is second difference matrix
            D = np.diff(np.eye(X_train_spline.shape[1]), n=2, axis=0)
            P = D.T @ D

            # Solve penalized least squares
            beta = np.linalg.solve(XtX + lambda_ * P + alpha_ * I, Xty)

            # Predictions for train, validation, and test sets
            train_preds.append(X_train_spline[:, :len(beta)] @ beta)
            val_preds.append(X_val_spline[:, :len(beta)] @ beta)
            test_preds.append(X_test_spline[:, :len(beta)] @ beta)
            coefficients.append(beta)

        except Exception as e:
            print(f"✗ Error at maturity {maturity}: {e}")
            return None, None  # Skip this maturity if an error occurs

    # Combine predictions for all maturities
    train_preds = np.column_stack(train_preds)
    val_preds = np.column_stack(val_preds)
    test_preds = np.column_stack(test_preds)

    # Create RMSE results dataframe
    rmse_results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [compute_rmse(train_y, train_preds),
                 compute_rmse(val_y, val_preds),
                 compute_rmse(test_y, test_preds)]
    }

    results_df = pd.DataFrame(rmse_results)

    # Convert coefficients to DataFrame
    coefficients_df = pd.DataFrame(coefficients).T
    coefficients_df.columns = [f"Coefficient_{i+1}" for i in range(coefficients_df.shape[1])]

    return results_df, coefficients_df

# Step 4: Grid search with rolling window CV for hyperparameters
def grid_search_ridge_smoothing(train_X, train_y, test_X, test_y, maturity_columns, lambda_range, alpha_range, knot_range, n_splits=5):
    best_config = None
    best_rmse = float('inf')
    best_rmse_results = None
    best_coefficients = None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_knots in knot_range:
        for lambda_ in lambda_range:
            for alpha_ in alpha_range:
                print(f"Testing λ={lambda_:.2e}, α={alpha_:.2e}, knots={n_knots}")
                val_rmse_scores = []

                # Rolling window CV
                for train_index, val_index in tscv.split(train_X):
                    X_train_cv, X_val_cv = train_X[train_index], train_X[val_index]
                    y_train_cv, y_val_cv = train_y[train_index], train_y[val_index]

                    # Fit ridge-regulated smoothing model
                    rmse_results, coefficients = fit_ridge_smoothing(lambda_, alpha_, n_knots, X_train_cv, y_train_cv,
                                                                     X_val_cv, y_val_cv, test_X, test_y, maturity_columns)
                    if rmse_results is None:
                        break  # Skip this configuration if any fold fails
                    val_rmse_scores.append(rmse_results['RMSE'][1])  # Get RMSE for validation set

                if len(val_rmse_scores) == n_splits:
                    avg_val_rmse = np.mean(val_rmse_scores)
                    if avg_val_rmse < best_rmse:
                        best_rmse = avg_val_rmse
                        best_config = (lambda_, alpha_, n_knots)
                        best_rmse_results = rmse_results
                        best_coefficients = coefficients

    print(f"\nBest config: λ={best_config[0]}, α={best_config[1]}, knots={best_config[2]}")
    print("\nRMSE for best config:")
    print(best_rmse_results)
    print("\nCoefficients for best config:")
    print(best_coefficients)

    return best_config, best_rmse_results, best_coefficients

# Hyperparameter ranges
lambda_range = np.logspace(-4, 4, 20)
alpha_range = np.logspace(-4, 4, 20)
knot_range = [3, 4, 5, 6, 7, 8, 9, 10]

# Call the grid search
best_config, best_rmse_results, best_coefficients = grid_search_ridge_smoothing(
    train_X, train_y, test_X, test_y,
    maturity_columns,
    lambda_range, alpha_range, knot_range,
    n_splits=5
)

################################################################################
#                 TRACKING DATA FRAME
################################################################################

#this section was copy pasted from our console

print("\nFor the Regular Quadratic Spline method, the optimal number of knots found based on a rolling window CV was 10")
print("\nFor the Smoothed Quadratic Spline method, the optimal hyperparameters found based on a rolling window CV were: \n λ=0.23357214690901212, knots=4")
print("\nFor the Smoothed Quadratic Spline method with added ridge penatlity, the optimal hyperparameters found based on a rolling window CV were: \n λ=0.615848211066026, α=1.623776739188721, knots=7\n")

# RMSE values from our results
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Quadratic Spline (QS)": [0.103281, 0.347961, 1.922895],
    "Smoothed QS": [0.184690, 0.169134, 0.911512],
    "Smoothed Ridge QS": [0.150609, 0.117274, 0.913927]
})

# Round values for clarity
rmse_table = rmse_table.round(4)

# Show the table
print(rmse_table)



################################################################################
#--------------------- CUBIC SPLINE METHODS     --------------------------------
################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Rolling Window CV for best number of knots ---
# window_size defines the number of observations in each rolling window.
# stride controls how far forward the window moves after each iteration (non-overlapping segments).
# spline_degree is the degree of the spline basis functions.
# knot_range defines how many internal knots to test in the spline basis.

window_size = 750
stride = 250
spline_degree = 3
knot_range = range(3, 11)  # Try 5 to 10 knots

def create_lagged_data(yields, lag=5): 
    """Create lagged predictor-response pairs with a specified lag (default 5 days)."""
    X = yields[:-lag]  # Shift the predictor by 'lag' days
    y = yields[lag:]   # Shift the target by 'lag' days
    return X, y

rmse_per_knot = {k: [] for k in knot_range}

for start in range(0, len(df_filtered) - window_size - 1, stride):
  
    end = start + window_size
    window = df_filtered.iloc[start:end]
    X, y = create_lagged_data(window[maturity_columns].values, lag=5)  # Pass the 5-day lag

    for n_knots in knot_range:
        val_errors = []
        for i in range(len(maturity_columns)):
            spline = SplineTransformer(degree=spline_degree, n_knots=n_knots, include_bias=False)
            X_spline = spline.fit_transform(X)
            coef, *_ = np.linalg.lstsq(X_spline, y[:, i], rcond=None)
            y_pred = X_spline @ coef
            val_errors.append(mean_squared_error(y[:, i], y_pred))
        avg_rmse = np.sqrt(np.mean(val_errors))
        rmse_per_knot[n_knots].append(avg_rmse)

# Average RMSE across all folds
avg_rmse_per_knot = {k: np.mean(v) for k, v in rmse_per_knot.items()}
best_k = min(avg_rmse_per_knot, key=avg_rmse_per_knot.get)
print(f"Best number of knots based on rolling CV: {best_k}")

# --- 4. Split final dataset ---
train_size = 0.7
val_size = 0.15
n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

train_X, train_y = create_lagged_data(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged_data(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged_data(test_df[maturity_columns].values, lag=5)

# --- 5. Train on best number of knots ---
models = {}
train_preds, val_preds, test_preds = [], [], []
coefficients = {}

for i, maturity in enumerate(maturity_columns):
    spline = SplineTransformer(degree=spline_degree, n_knots=best_k, include_bias=False)
    X_train_spline = spline.fit_transform(train_X)
    coef, *_ = np.linalg.lstsq(X_train_spline, train_y[:, i], rcond=None)

    models[maturity] = {'spline': spline, 'coef': coef}
    coefficients[maturity] = coef

    train_preds.append(X_train_spline @ coef)
    val_preds.append(spline.transform(val_X) @ coef)
    test_preds.append(spline.transform(test_X) @ coef)

train_preds = np.column_stack(train_preds)
val_preds = np.column_stack(val_preds)
test_preds = np.column_stack(test_preds)

# --- 6. RMSE Summary ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

rmse_results = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [
        compute_rmse(train_y, train_preds),
        compute_rmse(val_y, val_preds),
        compute_rmse(test_y, test_preds)
    ]
})
print("\nMSE Summary:")
print(rmse_results)

# --- 7. Coefficients Summary ---
coef_df = pd.DataFrame(coefficients).T
coef_df.columns = [f'Spline_{i+1}' for i in range(coef_df.shape[1])]
coef_df.index.name = "Maturity"
print("\nCoefficients from Spline Models (OLS):")
print(coef_df.round(4))

#-------------------------------------------------------------------------------
#                     Smoothing constraint
#-------------------------------------------------------------------------------

train_X, train_y = create_lagged_data(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged_data(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged_data(test_df[maturity_columns].values, lag=5)

# --- 1. Function to compute RMSE ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# --- 2. Function to fit smoothing spline and return RMSE on train/val/test along with coefficients ---
def fit_smoothing_spline(lambda_, n_knots, train_X, train_y, val_X, val_y, test_X, test_y, maturity_columns):
    train_preds, val_preds, test_preds = [], [], []
    coefficients = []

    for i, maturity in enumerate(maturity_columns):
        try:
            spline = SplineTransformer(degree=3, n_knots=n_knots, include_bias=False)
            X_train_spline = spline.fit_transform(train_X)
            X_val_spline = spline.transform(val_X)
            X_test_spline = spline.transform(test_X)
            
            # Compute second-difference penalty matrix.
            # Encourages adjacent spline coefficients to be similar (smoothness).

            n_basis = X_train_spline.shape[1]
            D = np.diff(np.eye(n_basis), n=2)
            P = D.T @ D

            XtX = X_train_spline.T @ X_train_spline
            Xty = X_train_spline.T @ train_y[:, i]

            # Align dimensions
            min_dim = min(P.shape[0], XtX.shape[0])
            P = P[:min_dim, :min_dim]
            XtX = XtX[:min_dim, :min_dim]
            Xty = Xty[:min_dim]

            beta = np.linalg.solve(XtX + lambda_ * P, Xty)

            train_preds.append(X_train_spline[:, :len(beta)] @ beta)
            val_preds.append(X_val_spline[:, :len(beta)] @ beta)
            test_preds.append(X_test_spline[:, :len(beta)] @ beta)
            coefficients.append(beta)  # Save the coefficients for this maturity

        except Exception as e:
            print(f"✗ Error at maturity {maturity}: {e}")
            return None, None  # Skip this config if it fails

    train_preds = np.column_stack(train_preds)
    val_preds = np.column_stack(val_preds)
    test_preds = np.column_stack(test_preds)

    # I will create a dataframe somewhat similar to the one before
    rmse_results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [compute_rmse(train_y, train_preds),
                 compute_rmse(val_y, val_preds),
                 compute_rmse(test_y, test_preds)]
    }

    # Convert RMSE results to DataFrame
    results_df = pd.DataFrame(rmse_results)

    # Convert coefficients to DataFrame
    coefficients_df = pd.DataFrame(coefficients).T  # Transpose to have one column per maturity
    coefficients_df.columns = [f"Coefficient_{i+1}" for i in range(coefficients_df.shape[1])]  # Rename columns

    return results_df, coefficients_df

# --- 3. Grid search with rolling window CV ---
def grid_search_smoothing(train_X, train_y, test_X, test_y, maturity_columns, lambda_range, knot_range, n_splits=5):
    best_config = None
    best_rmse = float('inf')
    best_rmse_results = None
    best_coefficients = None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_knots in knot_range:
        for lambda_ in lambda_range:
            print(f"Testing λ={lambda_:.2e}, knots={n_knots}")
            val_rmse_scores = []

            for train_index, val_index in tscv.split(train_X):
                X_train_cv, X_val_cv = train_X[train_index], train_X[val_index]
                y_train_cv, y_val_cv = train_y[train_index], train_y[val_index]

                rmse_results, coefficients = fit_smoothing_spline(lambda_, n_knots, X_train_cv, y_train_cv,
                                                                 X_val_cv, y_val_cv, test_X, test_y, maturity_columns)
                if rmse_results is None:
                    break  # Skip this config if any fold fails
                val_rmse_scores.append(rmse_results['RMSE'][1])  # Get the RMSE for validation as a scalar

            if len(val_rmse_scores) == n_splits:
                avg_val_rmse = np.mean(val_rmse_scores)
                if avg_val_rmse < best_rmse:
                    best_rmse = avg_val_rmse
                    best_config = (lambda_, n_knots)
                    best_rmse_results = rmse_results  # from last fold
                    best_coefficients = coefficients  # Store the coefficients for the best config

    print(f"\nTest config: λ={best_config[0]}, knots={best_config[1]}")
    print("\nRMSE for best config:")
    print(best_rmse_results)
    print("\nCoefficients for best config:")
    print(best_coefficients)
    return best_config, best_rmse_results, best_coefficients

lambda_range = np.logspace(-4, 4, 20)
knot_range = [3, 4, 5, 6, 7, 8, 9, 10]

best_config, best_rmse_results, best_coefficients = grid_search_smoothing(
    train_X, train_y, test_X, test_y,
    maturity_columns,
    lambda_range, knot_range,
    n_splits=5
)

#-------------------------------------------------------------------------------
#                             RIDGE + SMOOTHING
#-------------------------------------------------------------------------------
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

# Step 1: Create lagged data
def create_lagged_data(data, lag):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i])  # Previous 'lag' observations as features
        y.append(data[i])          # Current observation as target
    return np.array(X), np.array(y)

# Step 2: Compute RMSE
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# Step 3: Ridge + Smoothing function (with lambda, alpha, and knots)
def fit_ridge_smoothing(lambda_, alpha_, n_knots, train_X, train_y, val_X, val_y, test_X, test_y, maturity_columns):
    train_preds, val_preds, test_preds = [], [], []
    coefficients = []

    for i, maturity in enumerate(maturity_columns):
        try:
            # Apply smoothing spline transformation
            spline = SplineTransformer(degree=3, n_knots=n_knots, include_bias=False)
            X_train_spline = spline.fit_transform(train_X)
            X_val_spline = spline.transform(val_X)
            X_test_spline = spline.transform(test_X)
            
            # Precompute terms
            XtX = X_train_spline.T @ X_train_spline
            Xty = X_train_spline.T @ train_y[:, i]
            I = np.eye(XtX.shape[0])

            # Build smoothing penalty matrix P
            # P = D^T D where D is second difference matrix
            D = np.diff(np.eye(X_train_spline.shape[1]), n=2, axis=0)
            P = D.T @ D

            # Solve penalized least squares
            beta = np.linalg.solve(XtX + lambda_ * P + alpha_ * I, Xty)

            # Predictions for train, validation, and test sets
            train_preds.append(X_train_spline[:, :len(beta)] @ beta)
            val_preds.append(X_val_spline[:, :len(beta)] @ beta)
            test_preds.append(X_test_spline[:, :len(beta)] @ beta)
            coefficients.append(beta)

        except Exception as e:
            print(f"✗ Error at maturity {maturity}: {e}")
            return None, None  # Skip this maturity if an error occurs

    # Combine predictions for all maturities
    train_preds = np.column_stack(train_preds)
    val_preds = np.column_stack(val_preds)
    test_preds = np.column_stack(test_preds)

    # Create RMSE results dataframe
    rmse_results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [compute_rmse(train_y, train_preds),
                 compute_rmse(val_y, val_preds),
                 compute_rmse(test_y, test_preds)]
    }

    results_df = pd.DataFrame(rmse_results)

    # Convert coefficients to DataFrame
    coefficients_df = pd.DataFrame(coefficients).T
    coefficients_df.columns = [f"Coefficient_{i+1}" for i in range(coefficients_df.shape[1])]

    return results_df, coefficients_df

# Step 4: Grid search with rolling window CV for hyperparameters
def grid_search_ridge_smoothing(train_X, train_y, test_X, test_y, maturity_columns, lambda_range, alpha_range, knot_range, n_splits=5):
    best_config = None
    best_rmse = float('inf')
    best_rmse_results = None
    best_coefficients = None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_knots in knot_range:
        for lambda_ in lambda_range:
            for alpha_ in alpha_range:
                print(f"Testing λ={lambda_:.2e}, α={alpha_:.2e}, knots={n_knots}")
                val_rmse_scores = []

                # Rolling window CV
                for train_index, val_index in tscv.split(train_X):
                    X_train_cv, X_val_cv = train_X[train_index], train_X[val_index]
                    y_train_cv, y_val_cv = train_y[train_index], train_y[val_index]

                    # Fit ridge-regulated smoothing model
                    rmse_results, coefficients = fit_ridge_smoothing(lambda_, alpha_, n_knots, X_train_cv, y_train_cv,
                                                                     X_val_cv, y_val_cv, test_X, test_y, maturity_columns)
                    if rmse_results is None:
                        break  # Skip this configuration if any fold fails
                    val_rmse_scores.append(rmse_results['RMSE'][1])  # Get RMSE for validation set

                if len(val_rmse_scores) == n_splits:
                    avg_val_rmse = np.mean(val_rmse_scores)
                    if avg_val_rmse < best_rmse:
                        best_rmse = avg_val_rmse
                        best_config = (lambda_, alpha_, n_knots)
                        best_rmse_results = rmse_results
                        best_coefficients = coefficients

    print(f"\nBest config: λ={best_config[0]}, α={best_config[1]}, knots={best_config[2]}")
    print("\nRMSE for best config:")
    print(best_rmse_results)
    print("\nCoefficients for best config:")
    print(best_coefficients)

    return best_config, best_rmse_results, best_coefficients

# Hyperparameter ranges
lambda_range = np.logspace(-4, 4, 20)
alpha_range = np.logspace(-4, 4, 20)
knot_range = [3, 4, 5, 6, 7, 8, 9, 10]

# Call the grid search
best_config, best_rmse_results, best_coefficients = grid_search_ridge_smoothing(
    train_X, train_y, test_X, test_y,
    maturity_columns,
    lambda_range, alpha_range, knot_range,
    n_splits=5
)


################################################################################
#                 TRACKING DATA FRAME
################################################################################

#this section was copy pasted from our console, it is only a tracking component

print("\nFor the Regular Cubic Spline method, the optimal number of knots found based on a rolling window CV was 10")
print("\nFor the Smoothed Cubic Spline method, the optimal hyperparameters found based on a rolling window CV were: \n λ=0.012742749857031334, knots=3")
print("\nFor the Smoothed Cubic Spline method with added ridge penatlity, the optimal hyperparameters found based on a rolling window CV were: \n λ=1438.44988828766, α=0.0001, knots=10\n")

# RMSE values from our results
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Cubic Spline (CS)": [0.094425, 0.300657, 0.593346],
    "Smoothed CS": [0.112993, 0.126142, 0.293215],
    "Smoothed Ridge CS": [0.137244, 0.090847, 0.294733]
})


# Round values for clarity
rmse_table = rmse_table.round(4)

# Show the table
print(rmse_table)



################################################################################
#--------------------- NEURAL NETWORKS METHOD   --------------------------------
################################################################################

#-------------------------------------------------------------------------------
#                 STACKED LSTM WITH INCREASED DROPOUTS
##------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

#  Set random seeds for reproducibility ---
np.random.seed(43)
torch.manual_seed(43)

#  Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

#  Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

#. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

#  Create lagged sequences ---
def create_lagged(yield_data, lag=5):
    X = []
    y = []
    for i in range(lag, len(yield_data)):
        X.append(yield_data[i-lag:i])  # past 'lag' days
        y.append(yield_data[i])         # predict current day
    return np.array(X), np.array(y)

lag = 5  

train_X, train_y = create_lagged(train_df[maturity_columns].values, lag=lag)
val_X, val_y = create_lagged(val_df[maturity_columns].values, lag=lag)
test_X, test_y = create_lagged(test_df[maturity_columns].values, lag=lag)

#  Convert to PyTorch tensors ---
def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

train_X_tensor = to_tensor(train_X)
train_y_tensor = to_tensor(train_y)
val_X_tensor = to_tensor(val_X)
val_y_tensor = to_tensor(val_y)
test_X_tensor = to_tensor(test_X)
test_y_tensor = to_tensor(test_y)

batch_size = 32
train_loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size)

# . Define LSTM Model ---
class YieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates):
        super(YieldLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        self.fc = nn.Linear(hidden_sizes[2], input_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        x = x[:, -1, :]  # take the last time step output
        return self.fc(x)

input_size = len(maturity_columns)
hidden_sizes = [64, 128, 256]
dropout_rates = [0.1, 0.2, 0.3]

model = YieldLSTM(input_size, hidden_sizes, dropout_rates)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#  Train the model with early stopping ---
num_epochs = 100
patience = 5
best_val_rmse = float('inf')
epochs_no_improve = 0
train_rmse_history = []
val_rmse_history = []
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_rmse = np.sqrt(np.mean(train_losses))
    train_rmse_history.append(train_rmse)

    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor)
        val_loss = criterion(val_preds, val_y_tensor)
        val_rmse = torch.sqrt(val_loss).item()

    val_rmse_history.append(val_rmse)

    print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-5:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()
        epochs_no_improve = 0
        best_epoch = epoch
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

model.load_state_dict(best_model_state)

# . Evaluate model 
model.eval()
with torch.no_grad():
    train_preds = model(train_X_tensor)
    val_preds = model(val_X_tensor)
    test_preds = model(test_X_tensor)

train_rmse = np.sqrt(mean_squared_error(train_y, train_preds.numpy()))
val_rmse = np.sqrt(mean_squared_error(val_y, val_preds.numpy()))
test_rmse = np.sqrt(mean_squared_error(test_y, test_preds.numpy()))

train_mae = mean_absolute_error(train_y, train_preds.numpy())
val_mae = mean_absolute_error(val_y, val_preds.numpy())
test_mae = mean_absolute_error(test_y, test_preds.numpy())

results_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "RMSE": [train_rmse, val_rmse, test_rmse],
    "MAE": [train_mae, val_mae, test_mae]
})

print("\n--- RMSE and MAE Results ---")
print(results_df)

#  Plot validation RMSE over epochs ---
plt.figure(figsize=(8, 5))
plt.plot(val_rmse_history, marker='o', label='Validation RMSE', color='blue')
plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch (Early Stopping)')
plt.title('Validation RMSE per Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#  Print model parameters ---
print("\nModel Parameters by Layer:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} | Shape: {param.shape}")
        print(param.data)
        print("-" * 60)

#-------------------------------------------------------------------------------
#                           With Standarization
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

np.random.seed(43)
torch.manual_seed(43)

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

# Modify create_lagged function to incorporate a 5-day lag
def create_lagged(yield_data, lag=5):
    # Shift the yield data by the specified lag value
    X = yield_data[:-lag]  # All rows except the last 'lag' rows
    y = yield_data[lag:]   # All rows except the first 'lag' rows
    return X, y

# Apply the 5-day lag to your data
train_X, train_y = create_lagged(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged(test_df[maturity_columns].values, lag=5)

# --- Standardize based on training set ---
train_mean_X = train_X.mean(axis=0)
train_std_X = train_X.std(axis=0)
train_mean_y = train_y.mean(axis=0)
train_std_y = train_y.std(axis=0)

train_std_X[train_std_X == 0] = 1e-6    #this just to avoid division by zero
train_std_y[train_std_y == 0] = 1e-6    #this is also just to avoid division by zero

train_X_std = (train_X - train_mean_X) / train_std_X
train_y_std = (train_y - train_mean_y) / train_std_y
val_X_std = (val_X - train_mean_X) / train_std_X
val_y_std = (val_y - train_mean_y) / train_std_y
test_X_std = (test_X - train_mean_X) / train_std_X
test_y_std = (test_y - train_mean_y) / train_std_y

# Convert to tensors
train_X_tensor = to_tensor(train_X_std).unsqueeze(1)
train_y_tensor = to_tensor(train_y_std)
val_X_tensor = to_tensor(val_X_std).unsqueeze(1)
val_y_tensor = to_tensor(val_y_std)
test_X_tensor = to_tensor(test_X_std).unsqueeze(1)
test_y_tensor = to_tensor(test_y_std)

# Create DataLoader for training, validation, and testing
batch_size = 32
train_loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size)



class YieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates):
        super(YieldLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        self.fc = nn.Linear(hidden_sizes[2], input_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        x = x[:, -1, :]
        return self.fc(x)

input_size = len(maturity_columns)
hidden_sizes = [64, 128, 256]
dropout_rates = [0.1, 0.2, 0.3]

model = YieldLSTM(input_size, hidden_sizes, dropout_rates)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
patience = 5
best_val_rmse = float('inf')
epochs_no_improve = 0
train_rmse_history = []
val_rmse_history = []
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_rmse = np.sqrt(np.mean(train_losses))
    train_rmse_history.append(train_rmse)

    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor)
        val_loss = criterion(val_preds, val_y_tensor)
        val_rmse = torch.sqrt(val_loss).item()

    val_rmse_history.append(val_rmse)

    print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-5:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()
        epochs_no_improve = 0
        best_epoch = epoch
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

model.load_state_dict(best_model_state)

# --- Inverse transform predictions before calculating metrics ---
model.eval()
with torch.no_grad():
    train_preds_std = model(train_X_tensor).numpy()
    val_preds_std = model(val_X_tensor).numpy()
    test_preds_std = model(test_X_tensor).numpy()

# Inverse transform
train_preds = train_preds_std * train_std_y + train_mean_y
val_preds = val_preds_std * train_std_y + train_mean_y
test_preds = test_preds_std * train_std_y + train_mean_y

train_y_orig = train_y_std * train_std_y + train_mean_y
val_y_orig = val_y_std * train_std_y + train_mean_y
test_y_orig = test_y_std * train_std_y + train_mean_y

train_rmse = np.sqrt(mean_squared_error(train_y_orig, train_preds))
val_rmse = np.sqrt(mean_squared_error(val_y_orig, val_preds))
test_rmse = np.sqrt(mean_squared_error(test_y_orig, test_preds))

train_mae = mean_absolute_error(train_y_orig, train_preds)
val_mae = mean_absolute_error(val_y_orig, val_preds)
test_mae = mean_absolute_error(test_y_orig, test_preds)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "RMSE": [train_rmse, val_rmse, test_rmse],
    "MAE": [train_mae, val_mae, test_mae]
})

print("\n--- RMSE and MAE Results ---")
print(results_df)

plt.figure(figsize=(8, 5))
plt.plot(val_rmse_history, marker='o', label='Validation RMSE', color='blue')
plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch (Early Stopping)')
plt.title('Validation RMSE per Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nModel Parameters by Layer:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} | Shape: {param.shape}")
        print(param.data)
        print("-" * 60)

#-------------------------------------------------------------------------------
#                       Weights sharing
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

np.random.seed(43)
torch.manual_seed(43)

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

# Modify create_lagged function to incorporate a 5-day lag
def create_lagged(yield_data, lag=5):
    # Shift the yield data by the specified lag value
    X = yield_data[:-lag]  # All rows except the last 'lag' rows
    y = yield_data[lag:]   # All rows except the first 'lag' rows
    return X, y

# Apply the 5-day lag to your data
train_X, train_y = create_lagged(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged(test_df[maturity_columns].values, lag=5)

# --- Standardize based on training set ---
train_mean_X = train_X.mean(axis=0)
train_std_X = train_X.std(axis=0)
train_mean_y = train_y.mean(axis=0)
train_std_y = train_y.std(axis=0)

train_std_X[train_std_X == 0] = 1e-6    #this is to avoid division by zero during our 

train_std_y[train_std_y == 0] = 1e-6

train_X_std = (train_X - train_mean_X) / train_std_X
train_y_std = (train_y - train_mean_y) / train_std_y
val_X_std = (val_X - train_mean_X) / train_std_X
val_y_std = (val_y - train_mean_y) / train_std_y
test_X_std = (test_X - train_mean_X) / train_std_X
test_y_std = (test_y - train_mean_y) / train_std_y

# Convert to tensors
train_X_tensor = to_tensor(train_X_std).unsqueeze(1)
train_y_tensor = to_tensor(train_y_std)
val_X_tensor = to_tensor(val_X_std).unsqueeze(1)
val_y_tensor = to_tensor(val_y_std)
test_X_tensor = to_tensor(test_X_std).unsqueeze(1)
test_y_tensor = to_tensor(test_y_std)

# Create DataLoader for training, validation, and testing
batch_size = 32
train_loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size)



class YieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(YieldLSTM, self).__init__()
        # Shared LSTM weight
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Apply the shared LSTM weights for all layers
        x, _ = self.lstm(x)  # Pass through shared LSTM layer
        x = self.dropout(x)
        x = x[:, -1, :]  # Use the last hidden state
        return self.fc(x)

input_size = len(maturity_columns)
hidden_size = 128
dropout_rate = 0.2

model = YieldLSTM(input_size, hidden_size, dropout_rate)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
patience = 5
best_val_rmse = float('inf')
epochs_no_improve = 0
train_rmse_history = []
val_rmse_history = []
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_rmse = np.sqrt(np.mean(train_losses))
    train_rmse_history.append(train_rmse)

    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor)
        val_loss = criterion(val_preds, val_y_tensor)
        val_rmse = torch.sqrt(val_loss).item()

    val_rmse_history.append(val_rmse)

    print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-5:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()
        epochs_no_improve = 0
        best_epoch = epoch
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

model.load_state_dict(best_model_state)

# --- Inverse transform predictions before calculating metrics ---
model.eval()
with torch.no_grad():
    train_preds_std = model(train_X_tensor).numpy()
    val_preds_std = model(val_X_tensor).numpy()
    test_preds_std = model(test_X_tensor).numpy()

# Inverse transform
train_preds = train_preds_std * train_std_y + train_mean_y
val_preds = val_preds_std * train_std_y + train_mean_y
test_preds = test_preds_std * train_std_y + train_mean_y

train_y_orig = train_y_std * train_std_y + train_mean_y
val_y_orig = val_y_std * train_std_y + train_mean_y
test_y_orig = test_y_std * train_std_y + train_mean_y

train_rmse = np.sqrt(mean_squared_error(train_y_orig, train_preds))
val_rmse = np.sqrt(mean_squared_error(val_y_orig, val_preds))
test_rmse = np.sqrt(mean_squared_error(test_y_orig, test_preds))

train_mae = mean_absolute_error(train_y_orig, train_preds)
val_mae = mean_absolute_error(val_y_orig, val_preds)
test_mae = mean_absolute_error(test_y_orig, test_preds)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "RMSE": [train_rmse, val_rmse, test_rmse],
    "MAE": [train_mae, val_mae, test_mae]
})

print("\n--- RMSE and MAE Results ---")
print(results_df)

plt.figure(figsize=(8, 5))
plt.plot(val_rmse_history, marker='o', label='Validation RMSE', color='blue')
plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch (Early Stopping)')
plt.title('Validation RMSE per Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nModel Parameters by Layer:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} | Shape: {param.shape}")
        print(param.data)
        print("-" * 60)


#-------------------------------------------------------------------------------
#                           Result Tracking
#-------------------------------------------------------------------------------

# RMSE values from our results
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "LSTM + Dropouts (LD)": [0.106546, 0.276051, 0.391191],
    "Standarized LD (SLD)": [0.133452, 0.238676, 0.315596],
    "Weight sharing SLD": [0.117901, 0.171687, 0.198434]
})

# Round values for clarity
rmse_table = rmse_table.round(4)

print("\n RMSE Tracking table\n")
# Show the table
print(rmse_table)


# RMSE values from our results
mae_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "LSTM + Dropouts (LD)": [0.078209, 0.177892, 0.34963],
    "Standarized LD (SLD)": [0.096974, 0.180930, 0.256928],
    "Weight sharing SLD": [0.083457, 0.125171, 0.156208]
})

# Round values for clarity
mae_table = mae_table.round(4)

print("\n MAE Tracking table\n")
# Show the table
print(mae_table)


################################################################################
#--------------------- ALGORITHM COMPARISON     --------------------------------
################################################################################

import pandas as pd
import matplotlib.pyplot as plt

# RMSE values from our results
#degrees 4 to 6 omitted due to the very large test rmse
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Degree 1": [0.108525, 0.132891, 0.157781],
    "Degree 2": [0.098563 , 0.283143, 0.343971],
    "Degree 3": [0.084339, 1.124518, 2.022311],
    "QS": [0.103281, 0.347961, 1.922895],
    "SQS": [0.184690, 0.169134, 0.911512],
    "SRQS": [0.150609, 0.117274, 0.913927],
    "CS": [0.094425, 0.300657, 0.593346],
    "SCS": [0.112993, 0.126142, 0.293215],
    "SRCS": [0.137244, 0.090847, 0.294733],
    "LD": [0.106546, 0.276051, 0.391191],
    "SLD": [0.133452, 0.238676, 0.315596],
    "WSLD": [0.117901, 0.171687, 0.198434]
})


# Define the list of methods (column names to compare)
methods = rmse_table.columns[1:]  # Skip the first column 'Dataset'

# Extract the Test RMSE values for each method
test_rmse_values = [rmse_table.loc[rmse_table["Dataset"] == "Test", method].values[0] for method in methods]

# Create a bar chart
fig = plt.figure()
fig.suptitle('Algorithm Error Comparison , (Test RMSE, the lower the better)')
ax = fig.add_subplot(111)

# Plot the bar chart
bars = ax.bar(methods, test_rmse_values)

# Rotate the x-axis labels for better readability
ax.set_xticklabels(methods, rotation=90)

# Set axis labels
ax.set_xlabel('Methods')
ax.set_ylabel('Test RMSE')

# Annotate each bar with the RMSE value above it (rounded to 3 decimal places)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

# Adjust plot size and display
fig.set_size_inches(15,8)
plt.show()


################################################################################
################################################################################
#         This FOURTH section will BE ON SEMI-MONTHLY CAPTURE & FIVE-DAY LAG PREDICTION PROBLEM
#                   It corresponds to section 2 (semi-annual lag part only)
################################################################################
################################################################################


################################################################################
#--------------------- POLYNOMIAL REGRESSION -----------------------------------
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]

# Exclude 2020 and 2021
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]

# Drop missing values
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 2.5. Resample: First of month + Closest after/on 14th ---
df_filtered['Year'] = df_filtered['Date'].dt.year
df_filtered['Month'] = df_filtered['Date'].dt.month

def select_first_and_14th(group):
    # 1. First available date in month
    first = group.loc[group['Date'] == group['Date'].min()]
    
    # 2. Closest date ON or AFTER 14th (prefer later dates)
    after_14th = group[group['Date'].dt.day >= 14]
    if not after_14th.empty:
        # If there are dates >= 14th, pick the one closest to 14
        closest_14th = after_14th.iloc[(after_14th['Date'].dt.day - 14).abs().argsort()].head(1)
    else:
        # If no dates >= 14th, pick the last date available
        closest_14th = group.iloc[[-1]]
    
    return pd.concat([first, closest_14th])

semi_monthly_df = df_filtered.groupby(['Year', 'Month']).apply(select_first_and_14th).reset_index(drop=True)

# Drop helper columns
semi_monthly_df = semi_monthly_df.drop(columns=['Year', 'Month'])

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15  

n = len(semi_monthly_df)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

# Split data
train_df = semi_monthly_df.iloc[:train_end]
val_df = semi_monthly_df.iloc[train_end:val_end]
test_df = semi_monthly_df.iloc[val_end:]


# --- 4. Shift data for lagged values ---
lag = 5  # Define lag of 5 days

# Lagged yields for training, validation, and testing (shift by 5 days)
train_lagged = train_df[maturity_columns].shift(lag).dropna().values
val_lagged = val_df[maturity_columns].shift(lag).dropna().values
test_lagged = test_df[maturity_columns].shift(lag).dropna().values

# Define the next time step as the target (5 days after lag)
train_target = train_df[maturity_columns].iloc[lag:].values
val_target = val_df[maturity_columns].iloc[lag:].values
test_target = test_df[maturity_columns].iloc[lag:].values


# --- 5. Test Polynomial Degrees and Evaluate RMSE ---
train_rmse, val_rmse, test_rmse = [], [], []
best_model = None
best_degree = None

for degree in range(1, 7):  # Test polynomial degrees from 1 to 6
    poly = PolynomialFeatures(degree=degree)
    
    # Apply polynomial transformation to the lagged data
    train_poly = poly.fit_transform(train_lagged)
    val_poly = poly.transform(val_lagged)
    test_poly = poly.transform(test_lagged)
    
    # Train model
    model = LinearRegression()
    model.fit(train_poly, train_target)
    
    # Predict on training, validation, and test data
    train_preds = model.predict(train_poly)
    val_preds = model.predict(val_poly)
    test_preds = model.predict(test_poly)
    
    # Compute RMSE for each maturity
    train_rmse.append(np.sqrt(mean_squared_error(train_target, train_preds)))
    val_rmse.append(np.sqrt(mean_squared_error(val_target, val_preds)))
    test_rmse.append(np.sqrt(mean_squared_error(test_target, test_preds)))
    
    # Check if this model is the best so far based on validation RMSE
    if best_degree is None or val_rmse[degree - 1] < val_rmse[best_degree - 1]:
        best_degree = degree
        best_model = model
        best_poly = poly  # Save the best polynomial transformer for later use

# --- 6. Identify Best Polynomial Degree Based on Validation RMSE ---
print(f"Best Polynomial Degree: {best_degree}")

# --- 7. Plot RMSE for each polynomial degree ---
plt.figure(figsize=(10, 6))
plt.plot(range(1, 7), train_rmse, label='Train RMSE', marker='o')
plt.plot(range(1, 7), val_rmse, label='Validation RMSE', marker='o')
plt.plot(range(1, 7), test_rmse, label='Test RMSE', marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("RMSE for Polynomial Degrees (1-6) on Yield Curve Forecasting")
plt.legend()
plt.tight_layout()
plt.show()

# --- 8. Display RMSE Results ---
rmse_df = pd.DataFrame({
    'Polynomial Degree': range(1, 7),
    'Train RMSE': train_rmse,
    'Validation RMSE': val_rmse,
    'Test RMSE': test_rmse
})
print(rmse_df)

# --- 9. Extract and Display the Best Model's Coefficients ---
print("\nBest Model Coefficients:")
coefficients = best_model.coef_
intercept = best_model.intercept_

print(f"Intercept: {intercept}")
print(f"Coefficients for Polynomial Degree {best_degree}:")
print(coefficients)

# Maturity names (this should correspond to the number of rows in coefficients)
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']

# Polynomial degree (number of terms in your polynomial, based on the number of columns in coefficients)
poly_features = [f"Poly_{i}" for i in range(coefficients.shape[1])]

# Prepare a list to collect rows for the DataFrame
coef_list = []

# Loop through each maturity and its corresponding coefficients
for i, maturity in enumerate(maturity_columns):
    for j, poly_feature in enumerate(poly_features):
        coef_list.append({
            'Maturity': maturity,
            'Polynomial Feature': poly_feature,
            'Coefficient': coefficients[i][j]
        })

# Create the coefficients DataFrame
coef_df = pd.DataFrame(coef_list)

# Add the intercept for each maturity
intercept_df = pd.DataFrame({
    'Maturity': maturity_columns,
    'Intercept': intercept
})

# Merge the intercepts with the coefficients DataFrame
final_df = pd.merge(coef_df, intercept_df, on="Maturity", how="left")

# Display the final DataFrame
print(final_df)
#  Forecasting Next Observation for All Maturities ---
def forecast_next_day(current_rates, model, poly_transformer):
    """
    Forecast the next obs's rates based on current rates using the trained model and polynomial transformation.
    
    Parameters:
    current_rates - Array of current rates for all maturities
    model - Trained model (best model)
    poly_transformer - Polynomial transformer used in training
    
    Returns:
    Array of predicted rates for next obs
    """
    # Apply the polynomial transformation to the current rates (reshape and transform)
    current_rates_reshaped = current_rates.reshape(1, -1)
    current_poly = poly_transformer.transform(current_rates_reshaped)
    
    # Forecast using the trained model
    next_day_rates = model.predict(current_poly)
    return next_day_rates[0]

# Forecast using the last available data point from the test set
last_observed_rates = test_df[maturity_columns].iloc[-1].values
next_day_forecast = forecast_next_day(last_observed_rates, best_model, best_poly)

# Create a DataFrame for forecast comparison
forecast_df = pd.DataFrame({
    'Maturity': maturity_columns,
    'Current Rate': last_observed_rates,
    'Forecasted Rate': next_day_forecast,
    'Change': next_day_forecast - last_observed_rates
})

# Print forecast for the next day
print("\nNext Day Forecast:")
print(forecast_df)

################################################################################
#--------------------- QUADRATIC SPLINE METHODS --------------------------------
################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 2b. Keep only 1st of month and closest date after 14th ---
def select_dates(df):
    selected_rows = []
    grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])

    for (year, month), group in grouped:
        group = group.sort_values('Date')
        first_of_month = group.iloc[0]
        after_14th = group[group['Date'].dt.day >= 14]

        if not after_14th.empty:
            closest_after_14th = after_14th.iloc[0]
        else:
            closest_after_14th = group.iloc[-1]

        selected_rows.append(first_of_month)
        selected_rows.append(closest_after_14th)

    selected_df = pd.DataFrame(selected_rows).drop_duplicates().sort_values('Date').reset_index(drop=True)
    return selected_df

df_filtered = select_dates(df_filtered)

# --- 3. Rolling Window CV for best number of knots ---
window_size = 750
stride = 250
spline_degree = 2
knot_range = range(3, 11)

def create_lagged_data(yields, lag=5): 
    X = yields[:-lag]
    y = yields[lag:]
    return X, y

rmse_per_knot = {k: [] for k in knot_range}

for start in range(0, len(df_filtered) - window_size - 1, stride):
    end = start + window_size
    window = df_filtered.iloc[start:end]
    X, y = create_lagged_data(window[maturity_columns].values, lag=5)

    for n_knots in knot_range:
        val_errors = []
        for i in range(len(maturity_columns)):
            spline = SplineTransformer(degree=spline_degree, n_knots=n_knots, include_bias=False)
            X_spline = spline.fit_transform(X)
            coef, *_ = np.linalg.lstsq(X_spline, y[:, i], rcond=None)
            y_pred = X_spline @ coef
            val_errors.append(mean_squared_error(y[:, i], y_pred))
        avg_rmse = np.sqrt(np.mean(val_errors))
        rmse_per_knot[n_knots].append(avg_rmse)

# Average RMSE across all folds
avg_rmse_per_knot = {k: np.mean(v) for k, v in rmse_per_knot.items()}
best_k = min(avg_rmse_per_knot, key=avg_rmse_per_knot.get)
print(f"Best number of knots based on rolling CV: {best_k}")

# --- 4. Split final dataset ---
train_size = 0.7
val_size = 0.15
n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

train_X, train_y = create_lagged_data(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged_data(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged_data(test_df[maturity_columns].values, lag=5)

# --- 5. Train on best number of knots ---
models = {}
train_preds, val_preds, test_preds = [], [], []
coefficients = {}

for i, maturity in enumerate(maturity_columns):
    spline = SplineTransformer(degree=spline_degree, n_knots=best_k, include_bias=False)
    X_train_spline = spline.fit_transform(train_X)
    coef, *_ = np.linalg.lstsq(X_train_spline, train_y[:, i], rcond=None)

    models[maturity] = {'spline': spline, 'coef': coef}
    coefficients[maturity] = coef

    train_preds.append(X_train_spline @ coef)
    val_preds.append(spline.transform(val_X) @ coef)
    test_preds.append(spline.transform(test_X) @ coef)

train_preds = np.column_stack(train_preds)
val_preds = np.column_stack(val_preds)
test_preds = np.column_stack(test_preds)

# --- 6. RMSE Summary ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

rmse_results = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [
        compute_rmse(train_y, train_preds),
        compute_rmse(val_y, val_preds),
        compute_rmse(test_y, test_preds)
    ]
})
print("\nMSE Summary:")
print(rmse_results)

# --- 7. Coefficients Summary ---
coef_df = pd.DataFrame(coefficients).T
coef_df.columns = [f'Spline_{i+1}' for i in range(coef_df.shape[1])]
coef_df.index.name = "Maturity"
print("\nCoefficients from Spline Models (OLS):")
print(coef_df.round(4))


#-------------------------------------------------------------------------------
#                     Smoothing constraint
#-------------------------------------------------------------------------------

def select_dates(df):
    selected_rows = []
    grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])

    for (year, month), group in grouped:
        group = group.sort_values('Date')
        first_of_month = group.iloc[0]
        after_14th = group[group['Date'].dt.day >= 14]

        if not after_14th.empty:
            closest_after_14th = after_14th.iloc[0]
        else:
            closest_after_14th = group.iloc[-1]

        selected_rows.append(first_of_month)
        selected_rows.append(closest_after_14th)

    selected_df = pd.DataFrame(selected_rows).drop_duplicates().sort_values('Date').reset_index(drop=True)
    return selected_df

# --- 2. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 3. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 4. Apply date filtering ---
df_filtered = select_dates(df_filtered)

# --- 5. Split into Train, Validation, and Test Sets ---
train_size = 0.7
val_size = 0.15
n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

train_X, train_y = create_lagged_data(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged_data(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged_data(test_df[maturity_columns].values, lag=5)

# --- 6. Function to compute RMSE ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# --- 7. Function to fit smoothing spline and return RMSE on train/val/test along with coefficients ---
def fit_smoothing_spline(lambda_, n_knots, train_X, train_y, val_X, val_y, test_X, test_y, maturity_columns):
    train_preds, val_preds, test_preds = [], [], []
    coefficients = []

    for i, maturity in enumerate(maturity_columns):
        try:
            spline = SplineTransformer(degree=2, n_knots=n_knots, include_bias=False)
            X_train_spline = spline.fit_transform(train_X)
            X_val_spline = spline.transform(val_X)
            X_test_spline = spline.transform(test_X)
            
            # Compute second-difference penalty matrix.
            n_basis = X_train_spline.shape[1]
            D = np.diff(np.eye(n_basis), n=2)
            P = D.T @ D

            XtX = X_train_spline.T @ X_train_spline
            Xty = X_train_spline.T @ train_y[:, i]

            # Align dimensions
            min_dim = min(P.shape[0], XtX.shape[0])
            P = P[:min_dim, :min_dim]
            XtX = XtX[:min_dim, :min_dim]
            Xty = Xty[:min_dim]

            beta = np.linalg.solve(XtX + lambda_ * P, Xty)

            train_preds.append(X_train_spline[:, :len(beta)] @ beta)
            val_preds.append(X_val_spline[:, :len(beta)] @ beta)
            test_preds.append(X_test_spline[:, :len(beta)] @ beta)
            coefficients.append(beta)  # Save the coefficients for this maturity

        except Exception as e:
            print(f"✗ Error at maturity {maturity}: {e}")
            return None, None  # Skip this config if it fails

    train_preds = np.column_stack(train_preds)
    val_preds = np.column_stack(val_preds)
    test_preds = np.column_stack(test_preds)

    # I will create a dataframe somewhat similar to the one before
    rmse_results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [compute_rmse(train_y, train_preds),
                 compute_rmse(val_y, val_preds),
                 compute_rmse(test_y, test_preds)]
    }

    # Convert RMSE results to DataFrame
    results_df = pd.DataFrame(rmse_results)

    # Convert coefficients to DataFrame
    coefficients_df = pd.DataFrame(coefficients).T  # Transpose to have one column per maturity
    coefficients_df.columns = [f"Coefficient_{i+1}" for i in range(coefficients_df.shape[1])]  # Rename columns

    return results_df, coefficients_df

# --- 8. Grid search with rolling window CV ---
def grid_search_smoothing(train_X, train_y, test_X, test_y, maturity_columns, lambda_range, knot_range, n_splits=5):
    best_config = None
    best_rmse = float('inf')
    best_rmse_results = None
    best_coefficients = None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_knots in knot_range:
        for lambda_ in lambda_range:
            print(f"Testing λ={lambda_:.2e}, knots={n_knots}")
            val_rmse_scores = []

            for train_index, val_index in tscv.split(train_X):
                X_train_cv, X_val_cv = train_X[train_index], train_X[val_index]
                y_train_cv, y_val_cv = train_y[train_index], train_y[val_index]

                rmse_results, coefficients = fit_smoothing_spline(lambda_, n_knots, X_train_cv, y_train_cv,
                                                                 X_val_cv, y_val_cv, test_X, test_y, maturity_columns)
                if rmse_results is None:
                    break  # Skip this config if any fold fails
                val_rmse_scores.append(rmse_results['RMSE'][1])  # Get the RMSE for validation as a scalar

            if len(val_rmse_scores) == n_splits:
                avg_val_rmse = np.mean(val_rmse_scores)
                if avg_val_rmse < best_rmse:
                    best_rmse = avg_val_rmse
                    best_config = (lambda_, n_knots)
                    best_rmse_results = rmse_results  # from last fold
                    best_coefficients = coefficients  # Store the coefficients for the best config

    print(f"\nTest config: λ={best_config[0]}, knots={best_config[1]}")
    print("\nRMSE for best config:")
    print(best_rmse_results)
    print("\nCoefficients for best config:")
    print(best_coefficients)
    return best_config, best_rmse_results, best_coefficients

lambda_range = np.logspace(-4, 4, 20)
knot_range = [3, 4, 5, 6, 7, 8, 9, 10]

best_config, best_rmse_results, best_coefficients = grid_search_smoothing(
    train_X, train_y, test_X, test_y,
    maturity_columns,
    lambda_range, knot_range,
    n_splits=5
)

#-------------------------------------------------------------------------------
#                             RIDGE + SMOOTHING
#-------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]

# --- 2b. Keep only 1st of month and closest date after 14th ---
def select_dates(df):
    selected_rows = []
    grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])

    for (year, month), group in grouped:
        group = group.sort_values('Date')
        first_of_month = group.iloc[0]
        after_14th = group[group['Date'].dt.day >= 14]

        if not after_14th.empty:
            closest_after_14th = after_14th.iloc[0]
        else:
            closest_after_14th = group.iloc[-1]

        selected_rows.append(first_of_month)
        selected_rows.append(closest_after_14th)

    selected_df = pd.DataFrame(selected_rows).drop_duplicates().sort_values('Date').reset_index(drop=True)
    return selected_df

df_filtered = select_dates(df_filtered)

# --- 3. Create Lagged Data ---
def create_lagged_data(data, lag):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i])  # Previous 'lag' observations as features
        y.append(data[i])          # Current observation as target
    return np.array(X), np.array(y)

# --- 4. Compute RMSE ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# --- 5. Ridge + Smoothing function ---
def fit_ridge_smoothing(lambda_, alpha_, n_knots, train_X, train_y, val_X, val_y, test_X, test_y, maturity_columns):
    train_preds, val_preds, test_preds = [], [], []
    coefficients = []

    for i, maturity in enumerate(maturity_columns):
        try:
            # Apply smoothing spline transformation
            spline = SplineTransformer(degree=2, n_knots=n_knots, include_bias=False)
            X_train_spline = spline.fit_transform(train_X)
            X_val_spline = spline.transform(val_X)
            X_test_spline = spline.transform(test_X)
            
            # Precompute terms
            XtX = X_train_spline.T @ X_train_spline
            Xty = X_train_spline.T @ train_y[:, i]
            I = np.eye(XtX.shape[0])

            # Build smoothing penalty matrix P
            D = np.diff(np.eye(X_train_spline.shape[1]), n=2, axis=0)
            P = D.T @ D

            # Solve penalized least squares
            beta = np.linalg.solve(XtX + lambda_ * P + alpha_ * I, Xty)

            # Predictions for train, validation, and test sets
            train_preds.append(X_train_spline[:, :len(beta)] @ beta)
            val_preds.append(X_val_spline[:, :len(beta)] @ beta)
            test_preds.append(X_test_spline[:, :len(beta)] @ beta)
            coefficients.append(beta)

        except Exception as e:
            print(f"✗ Error at maturity {maturity}: {e}")
            return None, None  # Skip this maturity if an error occurs

    # Combine predictions for all maturities
    train_preds = np.column_stack(train_preds)
    val_preds = np.column_stack(val_preds)
    test_preds = np.column_stack(test_preds)

    # Create RMSE results dataframe
    rmse_results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [compute_rmse(train_y, train_preds),
                 compute_rmse(val_y, val_preds),
                 compute_rmse(test_y, test_preds)]
    }

    results_df = pd.DataFrame(rmse_results)

    # Convert coefficients to DataFrame
    coefficients_df = pd.DataFrame(coefficients).T
    coefficients_df.columns = [f"Coefficient_{i+1}" for i in range(coefficients_df.shape[1])]

    return results_df, coefficients_df

# --- 6. Grid search with rolling window CV for hyperparameters ---
def grid_search_ridge_smoothing(train_X, train_y, test_X, test_y, maturity_columns, lambda_range, alpha_range, knot_range, n_splits=5):
    best_config = None
    best_rmse = float('inf')
    best_rmse_results = None
    best_coefficients = None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_knots in knot_range:
        for lambda_ in lambda_range:
            for alpha_ in alpha_range:
                print(f"Testing λ={lambda_:.2e}, α={alpha_:.2e}, knots={n_knots}")
                val_rmse_scores = []

                # Rolling window CV
                for train_index, val_index in tscv.split(train_X):
                    X_train_cv, X_val_cv = train_X[train_index], train_X[val_index]
                    y_train_cv, y_val_cv = train_y[train_index], train_y[val_index]

                    # Fit ridge-regulated smoothing model
                    rmse_results, coefficients = fit_ridge_smoothing(lambda_, alpha_, n_knots, X_train_cv, y_train_cv,
                                                                     X_val_cv, y_val_cv, test_X, test_y, maturity_columns)
                    if rmse_results is None:
                        break  # Skip this configuration if any fold fails
                    val_rmse_scores.append(rmse_results['RMSE'][1])  # Get RMSE for validation set

                if len(val_rmse_scores) == n_splits:
                    avg_val_rmse = np.mean(val_rmse_scores)
                    if avg_val_rmse < best_rmse:
                        best_rmse = avg_val_rmse
                        best_config = (lambda_, alpha_, n_knots)
                        best_rmse_results = rmse_results
                        best_coefficients = coefficients

    print(f"\nBest config: λ={best_config[0]}, α={best_config[1]}, knots={best_config[2]}")
    print("\nRMSE for best config:")
    print(best_rmse_results)
    print("\nCoefficients for best config:")
    print(best_coefficients)

    return best_config, best_rmse_results, best_coefficients

# Hyperparameter ranges
lambda_range = np.logspace(-4, 4, 20)
alpha_range = np.logspace(-4, 4, 20)
knot_range = [3, 4, 5, 6, 7, 8, 9, 10]

# Call the grid search
best_config, best_rmse_results, best_coefficients = grid_search_ridge_smoothing(
    train_X, train_y, test_X, test_y,
    maturity_columns,
    lambda_range, alpha_range, knot_range,
    n_splits=5
)


################################################################################
#                 TRACKING DATA FRAME
################################################################################

#this section was copy pasted from our console

print("\nFor the Regular Quadratic Spline method, the optimal number of knots found based on a rolling window CV was 3")
print("\nFor the Smoothed Quadratic Spline method, the optimal hyperparameters found based on a rolling window CV were: \n λ=0.004832930238571752, knots=5")
print("\nFor the Smoothed Quadratic Spline method with added ridge penatlity, the optimal hyperparameters found based on a rolling window CV were: \n λ=1438.44988828766, α=0.0001, knots=9\n")

# RMSE values from our results
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Quadratic Spline (QS)": [0.262214, 0.661323, 4.231415],
    "Smoothed QS": [0.266433, 0.245838, 1.670834],
    "Smoothed Ridge QS": [0.398973, 0.316107, 0.722050]
})

# Round values for clarity
rmse_table = rmse_table.round(4)

# Show the table
print(rmse_table)


################################################################################
#--------------------- CUBIC SPLINE METHODS ------------------------------------
################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 2b. Keep only 1st of month and closest date after 14th ---
def select_dates(df):
    selected_rows = []
    grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])

    for (year, month), group in grouped:
        group = group.sort_values('Date')
        first_of_month = group.iloc[0]
        after_14th = group[group['Date'].dt.day >= 14]

        if not after_14th.empty:
            closest_after_14th = after_14th.iloc[0]
        else:
            closest_after_14th = group.iloc[-1]

        selected_rows.append(first_of_month)
        selected_rows.append(closest_after_14th)

    selected_df = pd.DataFrame(selected_rows).drop_duplicates().sort_values('Date').reset_index(drop=True)
    return selected_df

df_filtered = select_dates(df_filtered)

# --- 3. Rolling Window CV for best number of knots ---
window_size = 750
stride = 250
spline_degree = 3
knot_range = range(3, 11)

def create_lagged_data(yields, lag=5): 
    X = yields[:-lag]
    y = yields[lag:]
    return X, y

rmse_per_knot = {k: [] for k in knot_range}

for start in range(0, len(df_filtered) - window_size - 1, stride):
    end = start + window_size
    window = df_filtered.iloc[start:end]
    X, y = create_lagged_data(window[maturity_columns].values, lag=5)

    for n_knots in knot_range:
        val_errors = []
        for i in range(len(maturity_columns)):
            spline = SplineTransformer(degree=spline_degree, n_knots=n_knots, include_bias=False)
            X_spline = spline.fit_transform(X)
            coef, *_ = np.linalg.lstsq(X_spline, y[:, i], rcond=None)
            y_pred = X_spline @ coef
            val_errors.append(mean_squared_error(y[:, i], y_pred))
        avg_rmse = np.sqrt(np.mean(val_errors))
        rmse_per_knot[n_knots].append(avg_rmse)

# Average RMSE across all folds
avg_rmse_per_knot = {k: np.mean(v) for k, v in rmse_per_knot.items()}
best_k = min(avg_rmse_per_knot, key=avg_rmse_per_knot.get)
print(f"Best number of knots based on rolling CV: {best_k}")

# --- 4. Split final dataset ---
train_size = 0.7
val_size = 0.15
n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

train_X, train_y = create_lagged_data(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged_data(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged_data(test_df[maturity_columns].values, lag=5)

# --- 5. Train on best number of knots ---
models = {}
train_preds, val_preds, test_preds = [], [], []
coefficients = {}

for i, maturity in enumerate(maturity_columns):
    spline = SplineTransformer(degree=spline_degree, n_knots=best_k, include_bias=False)
    X_train_spline = spline.fit_transform(train_X)
    coef, *_ = np.linalg.lstsq(X_train_spline, train_y[:, i], rcond=None)

    models[maturity] = {'spline': spline, 'coef': coef}
    coefficients[maturity] = coef

    train_preds.append(X_train_spline @ coef)
    val_preds.append(spline.transform(val_X) @ coef)
    test_preds.append(spline.transform(test_X) @ coef)

train_preds = np.column_stack(train_preds)
val_preds = np.column_stack(val_preds)
test_preds = np.column_stack(test_preds)

# --- 6. RMSE Summary ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

rmse_results = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [
        compute_rmse(train_y, train_preds),
        compute_rmse(val_y, val_preds),
        compute_rmse(test_y, test_preds)
    ]
})
print("\nMSE Summary:")
print(rmse_results)

# --- 7. Coefficients Summary ---
coef_df = pd.DataFrame(coefficients).T
coef_df.columns = [f'Spline_{i+1}' for i in range(coef_df.shape[1])]
coef_df.index.name = "Maturity"
print("\nCoefficients from Spline Models (OLS):")
print(coef_df.round(4))


#-------------------------------------------------------------------------------
#                     Smoothing constraint
#-------------------------------------------------------------------------------

def select_dates(df):
    selected_rows = []
    grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])

    for (year, month), group in grouped:
        group = group.sort_values('Date')
        first_of_month = group.iloc[0]
        after_14th = group[group['Date'].dt.day >= 14]

        if not after_14th.empty:
            closest_after_14th = after_14th.iloc[0]
        else:
            closest_after_14th = group.iloc[-1]

        selected_rows.append(first_of_month)
        selected_rows.append(closest_after_14th)

    selected_df = pd.DataFrame(selected_rows).drop_duplicates().sort_values('Date').reset_index(drop=True)
    return selected_df

# --- 2. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 3. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 4. Apply date filtering ---
df_filtered = select_dates(df_filtered)

# --- 5. Split into Train, Validation, and Test Sets ---
train_size = 0.7
val_size = 0.15
n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

train_X, train_y = create_lagged_data(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged_data(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged_data(test_df[maturity_columns].values, lag=5)

# --- 6. Function to compute RMSE ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# --- 7. Function to fit smoothing spline and return RMSE on train/val/test along with coefficients ---
def fit_smoothing_spline(lambda_, n_knots, train_X, train_y, val_X, val_y, test_X, test_y, maturity_columns):
    train_preds, val_preds, test_preds = [], [], []
    coefficients = []

    for i, maturity in enumerate(maturity_columns):
        try:
            spline = SplineTransformer(degree=3, n_knots=n_knots, include_bias=False)
            X_train_spline = spline.fit_transform(train_X)
            X_val_spline = spline.transform(val_X)
            X_test_spline = spline.transform(test_X)
            
            # Compute second-difference penalty matrix.
            n_basis = X_train_spline.shape[1]
            D = np.diff(np.eye(n_basis), n=2)
            P = D.T @ D

            XtX = X_train_spline.T @ X_train_spline
            Xty = X_train_spline.T @ train_y[:, i]

            # Align dimensions
            min_dim = min(P.shape[0], XtX.shape[0])
            P = P[:min_dim, :min_dim]
            XtX = XtX[:min_dim, :min_dim]
            Xty = Xty[:min_dim]

            beta = np.linalg.solve(XtX + lambda_ * P, Xty)

            train_preds.append(X_train_spline[:, :len(beta)] @ beta)
            val_preds.append(X_val_spline[:, :len(beta)] @ beta)
            test_preds.append(X_test_spline[:, :len(beta)] @ beta)
            coefficients.append(beta)  # Save the coefficients for this maturity

        except Exception as e:
            print(f"✗ Error at maturity {maturity}: {e}")
            return None, None  # Skip this config if it fails

    train_preds = np.column_stack(train_preds)
    val_preds = np.column_stack(val_preds)
    test_preds = np.column_stack(test_preds)

    # I will create a dataframe somewhat similar to the one before
    rmse_results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [compute_rmse(train_y, train_preds),
                 compute_rmse(val_y, val_preds),
                 compute_rmse(test_y, test_preds)]
    }

    # Convert RMSE results to DataFrame
    results_df = pd.DataFrame(rmse_results)

    # Convert coefficients to DataFrame
    coefficients_df = pd.DataFrame(coefficients).T  # Transpose to have one column per maturity
    coefficients_df.columns = [f"Coefficient_{i+1}" for i in range(coefficients_df.shape[1])]  # Rename columns

    return results_df, coefficients_df

# --- 8. Grid search with rolling window CV ---
def grid_search_smoothing(train_X, train_y, test_X, test_y, maturity_columns, lambda_range, knot_range, n_splits=5):
    best_config = None
    best_rmse = float('inf')
    best_rmse_results = None
    best_coefficients = None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_knots in knot_range:
        for lambda_ in lambda_range:
            print(f"Testing λ={lambda_:.2e}, knots={n_knots}")
            val_rmse_scores = []

            for train_index, val_index in tscv.split(train_X):
                X_train_cv, X_val_cv = train_X[train_index], train_X[val_index]
                y_train_cv, y_val_cv = train_y[train_index], train_y[val_index]

                rmse_results, coefficients = fit_smoothing_spline(lambda_, n_knots, X_train_cv, y_train_cv,
                                                                 X_val_cv, y_val_cv, test_X, test_y, maturity_columns)
                if rmse_results is None:
                    break  # Skip this config if any fold fails
                val_rmse_scores.append(rmse_results['RMSE'][1])  # Get the RMSE for validation as a scalar

            if len(val_rmse_scores) == n_splits:
                avg_val_rmse = np.mean(val_rmse_scores)
                if avg_val_rmse < best_rmse:
                    best_rmse = avg_val_rmse
                    best_config = (lambda_, n_knots)
                    best_rmse_results = rmse_results  # from last fold
                    best_coefficients = coefficients  # Store the coefficients for the best config

    print(f"\nTest config: λ={best_config[0]}, knots={best_config[1]}")
    print("\nRMSE for best config:")
    print(best_rmse_results)
    print("\nCoefficients for best config:")
    print(best_coefficients)
    return best_config, best_rmse_results, best_coefficients

lambda_range = np.logspace(-4, 4, 20)
knot_range = [3, 4, 5, 6, 7, 8, 9, 10]

best_config, best_rmse_results, best_coefficients = grid_search_smoothing(
    train_X, train_y, test_X, test_y,
    maturity_columns,
    lambda_range, knot_range,
    n_splits=5
)

#-------------------------------------------------------------------------------
#                             RIDGE + SMOOTHING
#-------------------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]

# --- 2b. Keep only 1st of month and closest date after 14th ---
def select_dates(df):
    selected_rows = []
    grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])

    for (year, month), group in grouped:
        group = group.sort_values('Date')
        first_of_month = group.iloc[0]
        after_14th = group[group['Date'].dt.day >= 14]

        if not after_14th.empty:
            closest_after_14th = after_14th.iloc[0]
        else:
            closest_after_14th = group.iloc[-1]

        selected_rows.append(first_of_month)
        selected_rows.append(closest_after_14th)

    selected_df = pd.DataFrame(selected_rows).drop_duplicates().sort_values('Date').reset_index(drop=True)
    return selected_df

df_filtered = select_dates(df_filtered)

# --- 3. Create Lagged Data ---
def create_lagged_data(data, lag):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i])  # Previous 'lag' observations as features
        y.append(data[i])          # Current observation as target
    return np.array(X), np.array(y)

# --- 4. Compute RMSE ---
def compute_rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))

# --- 5. Ridge + Smoothing function ---
def fit_ridge_smoothing(lambda_, alpha_, n_knots, train_X, train_y, val_X, val_y, test_X, test_y, maturity_columns):
    train_preds, val_preds, test_preds = [], [], []
    coefficients = []

    for i, maturity in enumerate(maturity_columns):
        try:
            # Apply smoothing spline transformation
            spline = SplineTransformer(degree=3, n_knots=n_knots, include_bias=False)
            X_train_spline = spline.fit_transform(train_X)
            X_val_spline = spline.transform(val_X)
            X_test_spline = spline.transform(test_X)
            
            # Precompute terms
            XtX = X_train_spline.T @ X_train_spline
            Xty = X_train_spline.T @ train_y[:, i]
            I = np.eye(XtX.shape[0])

            # Build smoothing penalty matrix P
            D = np.diff(np.eye(X_train_spline.shape[1]), n=2, axis=0)
            P = D.T @ D

            # Solve penalized least squares
            beta = np.linalg.solve(XtX + lambda_ * P + alpha_ * I, Xty)

            # Predictions for train, validation, and test sets
            train_preds.append(X_train_spline[:, :len(beta)] @ beta)
            val_preds.append(X_val_spline[:, :len(beta)] @ beta)
            test_preds.append(X_test_spline[:, :len(beta)] @ beta)
            coefficients.append(beta)

        except Exception as e:
            print(f"✗ Error at maturity {maturity}: {e}")
            return None, None  # Skip this maturity if an error occurs

    # Combine predictions for all maturities
    train_preds = np.column_stack(train_preds)
    val_preds = np.column_stack(val_preds)
    test_preds = np.column_stack(test_preds)

    # Create RMSE results dataframe
    rmse_results = {
        'Dataset': ['Train', 'Validation', 'Test'],
        'RMSE': [compute_rmse(train_y, train_preds),
                 compute_rmse(val_y, val_preds),
                 compute_rmse(test_y, test_preds)]
    }

    results_df = pd.DataFrame(rmse_results)

    # Convert coefficients to DataFrame
    coefficients_df = pd.DataFrame(coefficients).T
    coefficients_df.columns = [f"Coefficient_{i+1}" for i in range(coefficients_df.shape[1])]

    return results_df, coefficients_df

# --- 6. Grid search with rolling window CV for hyperparameters ---
def grid_search_ridge_smoothing(train_X, train_y, test_X, test_y, maturity_columns, lambda_range, alpha_range, knot_range, n_splits=5):
    best_config = None
    best_rmse = float('inf')
    best_rmse_results = None
    best_coefficients = None

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_knots in knot_range:
        for lambda_ in lambda_range:
            for alpha_ in alpha_range:
                print(f"Testing λ={lambda_:.2e}, α={alpha_:.2e}, knots={n_knots}")
                val_rmse_scores = []

                # Rolling window CV
                for train_index, val_index in tscv.split(train_X):
                    X_train_cv, X_val_cv = train_X[train_index], train_X[val_index]
                    y_train_cv, y_val_cv = train_y[train_index], train_y[val_index]

                    # Fit ridge-regulated smoothing model
                    rmse_results, coefficients = fit_ridge_smoothing(lambda_, alpha_, n_knots, X_train_cv, y_train_cv,
                                                                     X_val_cv, y_val_cv, test_X, test_y, maturity_columns)
                    if rmse_results is None:
                        break  # Skip this configuration if any fold fails
                    val_rmse_scores.append(rmse_results['RMSE'][1])  # Get RMSE for validation set

                if len(val_rmse_scores) == n_splits:
                    avg_val_rmse = np.mean(val_rmse_scores)
                    if avg_val_rmse < best_rmse:
                        best_rmse = avg_val_rmse
                        best_config = (lambda_, alpha_, n_knots)
                        best_rmse_results = rmse_results
                        best_coefficients = coefficients

    print(f"\nBest config: λ={best_config[0]}, α={best_config[1]}, knots={best_config[2]}")
    print("\nRMSE for best config:")
    print(best_rmse_results)
    print("\nCoefficients for best config:")
    print(best_coefficients)

    return best_config, best_rmse_results, best_coefficients

# Hyperparameter ranges
lambda_range = np.logspace(-4, 4, 20)
alpha_range = np.logspace(-4, 4, 20)
knot_range = [3, 4, 5, 6, 7, 8, 9, 10]

# Call the grid search
best_config, best_rmse_results, best_coefficients = grid_search_ridge_smoothing(
    train_X, train_y, test_X, test_y,
    maturity_columns,
    lambda_range, alpha_range, knot_range,
    n_splits=5
)


################################################################################
#                 TRACKING DATA FRAME
################################################################################

#this section was copy pasted from our console

print("\nFor the Regular Cubic Spline method, the optimal number of knots found based on a rolling window CV was 3")
print("\nFor the Smoothed Cubic Spline method, the optimal hyperparameters found based on a rolling window CV were: \n λ=0.615848211066026, knots=3")
print("\nFor the Smoothed Cubic Spline method with added ridge penatlity, the optimal hyperparameters found based on a rolling window CV were: \n λ=206.913808111479, α=0.004832930238571752, knots=5\n")

# RMSE values from our results
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Quadratic Spline (QS)": [0.245335, 0.816359, 0.928472],
    "Smoothed QS": [0.338313, 0.375277, 0.769270],
    "Smoothed Ridge QS": [0.356715, 0.294150, 0.629345]
})

# Round values for clarity
rmse_table = rmse_table.round(4)

# Show the table
print(rmse_table)




################################################################################
#--------------------- NEURAL NETWORK  METHODS --------------------------------
################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(420)
torch.manual_seed(420)

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]

# Exclude 2020 and 2021
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]

# Drop missing values
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 2.5. Resample: First of month + Closest after/on 14th ---
df_filtered['Year'] = df_filtered['Date'].dt.year
df_filtered['Month'] = df_filtered['Date'].dt.month

def select_first_and_14th(group):
    first = group.loc[group['Date'] == group['Date'].min()]
    after_14th = group[group['Date'].dt.day >= 14]
    if not after_14th.empty:
        closest_14th = after_14th.iloc[(after_14th['Date'].dt.day - 14).abs().argsort()].head(1)
    else:
        closest_14th = group.iloc[[-1]]
    return pd.concat([first, closest_14th])

semi_monthly_df = df_filtered.groupby(['Year', 'Month']).apply(select_first_and_14th).reset_index(drop=True)

# Drop helper columns
semi_monthly_df = semi_monthly_df.drop(columns=['Year', 'Month'])

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15  

n = len(semi_monthly_df)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

# Split data
train_df = semi_monthly_df.iloc[:train_end]
val_df = semi_monthly_df.iloc[train_end:val_end]
test_df = semi_monthly_df.iloc[val_end:]

# --- 4. Create lagged sequences ---
def create_lagged(yield_data, lag=5):
    X = []
    y = []
    for i in range(lag, len(yield_data)):
        X.append(yield_data[i-lag:i])
        y.append(yield_data[i])
    return np.array(X), np.array(y)

lag = 5  

train_X, train_y = create_lagged(train_df[maturity_columns].values, lag=lag)
val_X, val_y = create_lagged(val_df[maturity_columns].values, lag=lag)
test_X, test_y = create_lagged(test_df[maturity_columns].values, lag=lag)

# --- 5. Convert to PyTorch tensors ---
def to_tensor(data):
    return torch.tensor(data, dtype=torch.float32)

train_X_tensor = to_tensor(train_X)
train_y_tensor = to_tensor(train_y)
val_X_tensor = to_tensor(val_X)
val_y_tensor = to_tensor(val_y)
test_X_tensor = to_tensor(test_X)
test_y_tensor = to_tensor(test_y)

batch_size = 32
train_loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size)

# --- 6. Define LSTM Model ---
class YieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates):
        super(YieldLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        self.fc = nn.Linear(hidden_sizes[2], input_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        x = x[:, -1, :]
        return self.fc(x)

input_size = len(maturity_columns)
hidden_sizes = [64, 128, 256]
dropout_rates = [0.1, 0.2, 0.3]

model = YieldLSTM(input_size, hidden_sizes, dropout_rates)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 7. Train the model with early stopping ---
num_epochs = 100
patience = 5
best_val_rmse = float('inf')
epochs_no_improve = 0
train_rmse_history = []
val_rmse_history = []
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_rmse = np.sqrt(np.mean(train_losses))
    train_rmse_history.append(train_rmse)

    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor)
        val_loss = criterion(val_preds, val_y_tensor)
        val_rmse = torch.sqrt(val_loss).item()

    val_rmse_history.append(val_rmse)

    print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-5:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()
        epochs_no_improve = 0
        best_epoch = epoch
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

model.load_state_dict(best_model_state)

# --- 8. Evaluate model ---
model.eval()
with torch.no_grad():
    train_preds = model(train_X_tensor)
    val_preds = model(val_X_tensor)
    test_preds = model(test_X_tensor)

train_rmse = np.sqrt(mean_squared_error(train_y, train_preds.numpy()))
val_rmse = np.sqrt(mean_squared_error(val_y, val_preds.numpy()))
test_rmse = np.sqrt(mean_squared_error(test_y, test_preds.numpy()))

train_mae = mean_absolute_error(train_y, train_preds.numpy())
val_mae = mean_absolute_error(val_y, val_preds.numpy())
test_mae = mean_absolute_error(test_y, test_preds.numpy())

results_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "RMSE": [train_rmse, val_rmse, test_rmse],
    "MAE": [train_mae, val_mae, test_mae]
})

print("\n--- RMSE and MAE Results ---")
print(results_df)

# --- 9. Plot validation RMSE over epochs ---
plt.figure(figsize=(8, 5))
plt.plot(val_rmse_history, marker='o', label='Validation RMSE', color='blue')
plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch (Early Stopping)')
plt.title('Validation RMSE per Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 10. Print model parameters ---
print("\nModel Parameters by Layer:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} | Shape: {param.shape}")
        print(param.data)
        print("-" * 60)
        

# --- After training: Load the best model ---
model.load_state_dict(best_model_state)
model.eval()

# --- 8. Forecast the next observation ---

# Take the last 'lag' observations from the WHOLE dataset (train + val + test)
full_data = semi_monthly_df[maturity_columns].values
last_lagged_sequence = full_data[-lag:]  # shape (lag, num_maturities)

# Convert to tensor and add batch dimension
last_lagged_tensor = torch.tensor(last_lagged_sequence, dtype=torch.float32).unsqueeze(0)  # shape (1, lag, num_maturities)

# Predict the next yield curve
with torch.no_grad():
    next_pred = model(last_lagged_tensor)

# Convert to numpy array
next_pred_np = next_pred.squeeze(0).numpy()

# Display the prediction
predicted_yield_curve = pd.Series(next_pred_np, index=maturity_columns)
print("\nPredicted next yield curve:")
print(predicted_yield_curve)


#-------------------------------------------------------------------------------
#                           With Standarization
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

np.random.seed(420)
torch.manual_seed(420)

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 2.5. Resample: First date + Closest after/on 14th ---
df_filtered['Year'] = df_filtered['Date'].dt.year
df_filtered['Month'] = df_filtered['Date'].dt.month

def select_first_and_14th(group):
    first = group.loc[group['Date'] == group['Date'].min()]
    after_14th = group[group['Date'].dt.day >= 14]
    if not after_14th.empty:
        closest_14th = after_14th.iloc[(after_14th['Date'].dt.day - 14).abs().argsort()].head(1)
    else:
        closest_14th = group.iloc[[-1]]
    return pd.concat([first, closest_14th])

df_filtered = df_filtered.groupby(['Year', 'Month']).apply(select_first_and_14th).reset_index(drop=True)

# Drop helper columns
df_filtered = df_filtered.drop(columns=['Year', 'Month'])

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

# --- 4. Create lagged sequences ---
def create_lagged(yield_data, lag=5):
    X = yield_data[:-lag]
    y = yield_data[lag:]
    return X, y

train_X, train_y = create_lagged(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged(test_df[maturity_columns].values, lag=5)

# --- 5. Standardize based on training set ---
train_mean_X = train_X.mean(axis=0)
train_std_X = train_X.std(axis=0)
train_mean_y = train_y.mean(axis=0)
train_std_y = train_y.std(axis=0)

train_std_X[train_std_X == 0] = 1e-6
train_std_y[train_std_y == 0] = 1e-6

train_X_std = (train_X - train_mean_X) / train_std_X
train_y_std = (train_y - train_mean_y) / train_std_y
val_X_std = (val_X - train_mean_X) / train_std_X
val_y_std = (val_y - train_mean_y) / train_std_y
test_X_std = (test_X - train_mean_X) / train_std_X
test_y_std = (test_y - train_mean_y) / train_std_y

# --- 6. Convert to tensors ---
def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

train_X_tensor = to_tensor(train_X_std).unsqueeze(1)
train_y_tensor = to_tensor(train_y_std)
val_X_tensor = to_tensor(val_X_std).unsqueeze(1)
val_y_tensor = to_tensor(val_y_std)
test_X_tensor = to_tensor(test_X_std).unsqueeze(1)
test_y_tensor = to_tensor(test_y_std)

batch_size = 32
train_loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size)

# --- 7. Define LSTM Model ---
class YieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates):
        super(YieldLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rates[2])
        self.fc = nn.Linear(hidden_sizes[2], input_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        x = x[:, -1, :]
        return self.fc(x)

input_size = len(maturity_columns)
hidden_sizes = [64, 128, 256]
dropout_rates = [0.1, 0.2, 0.3]

model = YieldLSTM(input_size, hidden_sizes, dropout_rates)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 8. Train the model ---
num_epochs = 100
patience = 5
best_val_rmse = float('inf')
epochs_no_improve = 0
train_rmse_history = []
val_rmse_history = []
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_rmse = np.sqrt(np.mean(train_losses))
    train_rmse_history.append(train_rmse)

    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor)
        val_loss = criterion(val_preds, val_y_tensor)
        val_rmse = torch.sqrt(val_loss).item()

    val_rmse_history.append(val_rmse)

    print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-5:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()
        epochs_no_improve = 0
        best_epoch = epoch
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

model.load_state_dict(best_model_state)

# --- 9. Inverse transform and Evaluate ---
model.eval()
with torch.no_grad():
    train_preds_std = model(train_X_tensor).numpy()
    val_preds_std = model(val_X_tensor).numpy()
    test_preds_std = model(test_X_tensor).numpy()

train_preds = train_preds_std * train_std_y + train_mean_y
val_preds = val_preds_std * train_std_y + train_mean_y
test_preds = test_preds_std * train_std_y + train_mean_y

train_y_orig = train_y_std * train_std_y + train_mean_y
val_y_orig = val_y_std * train_std_y + train_mean_y
test_y_orig = test_y_std * train_std_y + train_mean_y

train_rmse = np.sqrt(mean_squared_error(train_y_orig, train_preds))
val_rmse = np.sqrt(mean_squared_error(val_y_orig, val_preds))
test_rmse = np.sqrt(mean_squared_error(test_y_orig, test_preds))

train_mae = mean_absolute_error(train_y_orig, train_preds)
val_mae = mean_absolute_error(val_y_orig, val_preds)
test_mae = mean_absolute_error(test_y_orig, test_preds)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "RMSE": [train_rmse, val_rmse, test_rmse],
    "MAE": [train_mae, val_mae, test_mae]
})

print("\n--- RMSE and MAE Results ---")
print(results_df)

plt.figure(figsize=(8, 5))
plt.plot(val_rmse_history, marker='o', label='Validation RMSE', color='blue')
plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch (Early Stopping)')
plt.title('Validation RMSE per Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nModel Parameters by Layer:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} | Shape: {param.shape}")
        print(param.data)
        print("-" * 60)

#-------------------------------------------------------------------------------
#                       Weights sharing with standarization
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

np.random.seed(420)
torch.manual_seed(420)

# --- 1. Load and clean data ---
df = pd.read_excel(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df_sorted = df.sort_values(by='Date')

# --- 2. Filter data ---
maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
df_filtered = df_sorted[['Date'] + maturity_columns]
df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]
df_filtered = df_filtered.dropna().reset_index(drop=True)

# --- 2.5 Filter to keep first day + closest to 14th day per month ---
def select_dates(df):
    selected_rows = []
    grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])

    for (year, month), group in grouped:
        group = group.sort_values('Date')
        first_day = group.iloc[0]

        # Find the date closest to the 14th
        group['days_diff'] = (group['Date'] - pd.Timestamp(year=year, month=month, day=14)).dt.days
        group['abs_diff'] = group['days_diff'].abs()

        # Prefer later dates if there is a tie (i.e., days_diff >= 0 is preferred)
        min_abs_diff = group['abs_diff'].min()
        candidates = group[group['abs_diff'] == min_abs_diff]
        if len(candidates) > 1:
            candidates = candidates[candidates['days_diff'] >= 0]
        if len(candidates) == 0:
            closest_day = group.loc[group['abs_diff'].idxmin()]
        else:
            closest_day = candidates.iloc[0]

        selected_rows.append(first_day)
        if closest_day['Date'] != first_day['Date']:
            selected_rows.append(closest_day)

    selected_df = pd.DataFrame(selected_rows).sort_values('Date').reset_index(drop=True)
    return selected_df.drop(columns=['days_diff', 'abs_diff'], errors='ignore')

df_filtered = select_dates(df_filtered)

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

# Helper function: tensor conversion
def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

# Modify create_lagged function to incorporate a 5-day lag
def create_lagged(yield_data, lag=5):
    X = yield_data[:-lag]  # All rows except the last 'lag' rows
    y = yield_data[lag:]   # All rows except the first 'lag' rows
    return X, y

# Apply the 5-day lag to your data
train_X, train_y = create_lagged(train_df[maturity_columns].values, lag=5)
val_X, val_y = create_lagged(val_df[maturity_columns].values, lag=5)
test_X, test_y = create_lagged(test_df[maturity_columns].values, lag=5)

# --- Standardize based on training set ---
train_mean_X = train_X.mean(axis=0)
train_std_X = train_X.std(axis=0)
train_mean_y = train_y.mean(axis=0)
train_std_y = train_y.std(axis=0)

train_std_X[train_std_X == 0] = 1e-6
train_std_y[train_std_y == 0] = 1e-6

train_X_std = (train_X - train_mean_X) / train_std_X
train_y_std = (train_y - train_mean_y) / train_std_y
val_X_std = (val_X - train_mean_X) / train_std_X
val_y_std = (val_y - train_mean_y) / train_std_y
test_X_std = (test_X - train_mean_X) / train_std_X
test_y_std = (test_y - train_mean_y) / train_std_y

# Convert to tensors
train_X_tensor = to_tensor(train_X_std).unsqueeze(1)
train_y_tensor = to_tensor(train_y_std)
val_X_tensor = to_tensor(val_X_std).unsqueeze(1)
val_y_tensor = to_tensor(val_y_std)
test_X_tensor = to_tensor(test_X_std).unsqueeze(1)
test_y_tensor = to_tensor(test_y_std)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(test_X_tensor, test_y_tensor), batch_size=batch_size)

# --- 4. Define Model ---
class YieldLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(YieldLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        return self.fc(x)

input_size = len(maturity_columns)
hidden_size = 128
dropout_rate = 0.2

model = YieldLSTM(input_size, hidden_size, dropout_rate)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 5. Train model ---
num_epochs = 100
patience = 5
best_val_rmse = float('inf')
epochs_no_improve = 0
train_rmse_history = []
val_rmse_history = []
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    train_rmse = np.sqrt(np.mean(train_losses))
    train_rmse_history.append(train_rmse)

    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_tensor)
        val_loss = criterion(val_preds, val_y_tensor)
        val_rmse = torch.sqrt(val_loss).item()

    val_rmse_history.append(val_rmse)

    print(f"Epoch {epoch+1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

    if val_rmse < best_val_rmse - 1e-5:
        best_val_rmse = val_rmse
        best_model_state = model.state_dict()
        epochs_no_improve = 0
        best_epoch = epoch
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break

model.load_state_dict(best_model_state)

# --- 6. Evaluate model ---
model.eval()
with torch.no_grad():
    train_preds_std = model(train_X_tensor).numpy()
    val_preds_std = model(val_X_tensor).numpy()
    test_preds_std = model(test_X_tensor).numpy()

# Inverse transform
train_preds = train_preds_std * train_std_y + train_mean_y
val_preds = val_preds_std * train_std_y + train_mean_y
test_preds = test_preds_std * train_std_y + train_mean_y

train_y_orig = train_y_std * train_std_y + train_mean_y
val_y_orig = val_y_std * train_std_y + train_mean_y
test_y_orig = test_y_std * train_std_y + train_mean_y

train_rmse = np.sqrt(mean_squared_error(train_y_orig, train_preds))
val_rmse = np.sqrt(mean_squared_error(val_y_orig, val_preds))
test_rmse = np.sqrt(mean_squared_error(test_y_orig, test_preds))

train_mae = mean_absolute_error(train_y_orig, train_preds)
val_mae = mean_absolute_error(val_y_orig, val_preds)
test_mae = mean_absolute_error(test_y_orig, test_preds)

results_df = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "RMSE": [train_rmse, val_rmse, test_rmse],
    "MAE": [train_mae, val_mae, test_mae]
})

print("\n--- RMSE and MAE Results ---")
print(results_df)

# --- 7. Plot Validation RMSE ---
plt.figure(figsize=(8, 5))
plt.plot(val_rmse_history, marker='o', label='Validation RMSE', color='blue')
plt.axvline(x=best_epoch, color='red', linestyle='--', label='Best Epoch (Early Stopping)')
plt.title('Validation RMSE per Epoch')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 8. Print model parameters ---
print("\nModel Parameters by Layer:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} | Shape: {param.shape}")
        print(param.data)
        print("-" * 60)


#-------------------------------------------------------------------------------
#                           Result Tracking
#-------------------------------------------------------------------------------

# RMSE values from our results
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "LSTM + Dropouts (LD)": [0.282381, 0.501207, 0.533319],
    "Standarized LD (SLD)": [0.508772, 0.843177, 0.875868],
    "Weight sharing SLD": [0.349353, 0.497099, 0.559976]
})

# Round values for clarity
rmse_table = rmse_table.round(4)

print("\n RMSE Tracking table\n")
# Show the table
print(rmse_table)



# RMSE values from our results
mae_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "LSTM + Dropouts (LD)": [0.209048, 0.400798, 0.443609],
    "Standarized LD (SLD)": [0.400053, 0.718613, 0.728529],
    "Weight sharing SLD": [0.249194, 0.385205, 0.448104]
})

# Round values for clarity
mae_table = mae_table.round(4)

print("\n MAE Tracking table\n")
# Show the table
print(mae_table)


################################################################################
#--------------------- ALGORITHM COMPARISON --------------------------------
################################################################################
import pandas as pd
import matplotlib.pyplot as plt

# RMSE values from our results
#degrees 3 to 6 omitted due to the very large test rmse
rmse_table = pd.DataFrame({
    "Dataset": ["Train", "Validation", "Test"],
    "Degree 1": [0.3081431, 0.527379, 0.572093],
    "Degree 2": [0.1917033 , 1.482915, 1.859050],
    "QS": [0.262214, 0.661323, 4.231415],
    "SQS": [0.266433, 0.245838, 1.670834],
    "SRQS": [0.398973, 0.316107, 0.722050],
    "CS": [0.245335, 0.816359, 0.928472],
    "SCS": [0.338313, 0.375277, 0.769270],
    "SRCS": [0.356715, 0.294150, 0.629345],
    "LD": [0.282381, 0.501207, 0.533319],
    "SLD": [0.508772, 0.843177, 0.875868],
    "WSLD": [0.349353, 0.497099, 0.559976]
})


# Define the list of methods (column names to compare)
methods = rmse_table.columns[1:]  # Skip the first column 'Dataset'

# Extract the Test RMSE values for each method
test_rmse_values = [rmse_table.loc[rmse_table["Dataset"] == "Test", method].values[0] for method in methods]

# Create a bar chart
fig = plt.figure()
fig.suptitle('Algorithm Error Comparison , (Test RMSE, the lower the better)')
ax = fig.add_subplot(111)

# Plot the bar chart
bars = ax.bar(methods, test_rmse_values)

# Rotate the x-axis labels for better readability
ax.set_xticklabels(methods, rotation=90)

# Set axis labels
ax.set_xlabel('Methods')
ax.set_ylabel('Test RMSE')

# Annotate each bar with the RMSE value above it (rounded to 3 decimal places)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=10)

# Adjust plot size and display
fig.set_size_inches(15,8)
plt.show()





################################################################################
################################################################################
#         This FIFTH section will  ON WHAT CORRESPONDS TO SECTION 3
#
################################################################################
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
from google.colab import drive
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import pandas as pd


# Mount Google Drive
drive.mount('/content/drive')

# --- 1. Load and clean data ---
file_path = "/content/drive/MyDrive/USTreasuryRates.xlsx" 

try:
  df = pd.read_excel(file_path)
  df['Date'] = pd.to_datetime(df['Date'])
  df_sorted = df.sort_values(by='Date')

  # --- 2. Filter data ---
  maturity_columns = ['6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
  df_filtered = df_sorted[['Date'] + maturity_columns]

  # Exclude 2020 and 2021
  df_filtered = df_filtered[~df_filtered['Date'].dt.year.isin([2020, 2021])]

  # Drop missing values
  df_filtered = df_filtered.dropna().reset_index(drop=True)

  print(df_filtered.head()) #check if the import is successful

except FileNotFoundError:
  print(f"Error: File not found at {file_path}. Please check the file path.")
except Exception as e:
  print(f"An error occurred: {e}")

# --- 3. Define split sizes ---
train_size = 0.7
val_size = 0.15
test_size = 0.15

n = len(df_filtered)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

# Split data
train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]

# --- 4. Convert maturities to numerical values (in years) ---
maturity_map = {
    '6 Mo': 0.5,
    '1 Yr': 1,
    '2 Yr': 2,
    '3 Yr': 3,
    '5 Yr': 5,
    '7 Yr': 7,
    '10 Yr': 10,
    '20 Yr': 20,
    '30 Yr': 30
}
maturities = [maturity_map[col] for col in maturity_columns]
tau = np.array(maturities)  # maturities as array for x-axis

# --- 5. Prepare yield matrices and date arrays ---
Y_train = train_df[maturity_columns].to_numpy()
Y_val = val_df[maturity_columns].to_numpy()
Y_test = test_df[maturity_columns].to_numpy()

dates_train = train_df['Date'].to_numpy()
dates_val = val_df['Date'].to_numpy()
dates_test = test_df['Date'].to_numpy()

train_df = df_filtered.iloc[:train_end]
val_df = df_filtered.iloc[train_end:val_end]
test_df = df_filtered.iloc[val_end:]


def create_training_data(df, lags, forecast_horizon):
    """
    Creates training data with lagged features and a future target.

    Args:
        df: DataFrame with 'Date' and maturity columns.
        lags: Number of past days to include as features.
        forecast_horizon: Number of days into the future to predict.

    Returns:
        Tuple: (X, y) where X is a NumPy array of lagged features and y is a NumPy array of future targets.
              Returns None if there's an error or insufficient data.
    """
    try:
        # Ensure the 'Date' column is a datetime index
        df = df.set_index('Date')

        # Create lagged features
        X = []
        y = []
        for i in range(lags, len(df) - forecast_horizon):  # Adjust loop for forecast_horizon
            X.append(df.iloc[i - lags:i].values.flatten())  # Flatten the multi-index
            y.append(df.iloc[i + forecast_horizon].values)  # Target is at forecast_horizon

        X = np.array(X)
        y = np.array(y)
        return X, y

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# create training, validation, and test sets with the following vonmbinations of lags and forecast horizon: lags = [10,30,60,90], forecast_horizons = [30,60,90]

lags = [10, 30, 60]
forecast_horizons = [1,5,10,30,60]

# Create datasets for different lags and forecast horizons
for lag in lags:
    for horizon in forecast_horizons:
        print(f"Creating dataset for lag={lag}, horizon={horizon}")

        # Create training data
        X_train, y_train = create_training_data(train_df.copy(), lag, horizon)
        if X_train is None:
            continue # Skip to the next iteration if data creation fails

        # Create validation data
        X_val, y_val = create_training_data(val_df.copy(), lag, horizon)
        if X_val is None:
            continue

        # Create test data
        X_test, y_test = create_training_data(test_df.copy(), lag, horizon)
        if X_test is None:
            continue

        # Now you have X_train, y_train, X_val, y_val, X_test, y_test for the current lag and horizon.
        # Save or process them as needed.  Example:
        np.savez(f"/content/drive/MyDrive/dataset_lag{lag}_horizon{horizon}.npz",
                 X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val,
                 X_test=X_test, y_test=y_test)

        print(f"Saved dataset for lag={lag}, horizon={horizon}")



def normalized_rmse(y_true, y_pred):
    """
    Calculates the normalized RMSE, taking volatility into account.
    """
    # Calculate the standard deviation of the true values (volatility)
    volatility = np.std(y_true)

    # Avoid division by zero if volatility is zero
    if volatility == 0:
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Normalize the RMSE by the volatility
    normalized_rmse = rmse / volatility

    return normalized_rmse





models = {
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet()
}

results = []

for lag in lags:
    for horizon in forecast_horizons:
        print(f"Evaluating models for lag={lag}, horizon={horizon}")
        try:
            data = np.load(f"/content/drive/MyDrive/dataset_lag{lag}_horizon{horizon}.npz")
            X_train, y_train = data["X_train"], data["y_train"]
            X_val, y_val = data["X_val"], data["y_val"]
            X_test, y_test = data["X_test"], data["y_test"]


            for model_name, model in models.items():
                print(f"Training {model_name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                nrmse = normalized_rmse(y_test, y_pred)
                results.append({
                    "model": model_name,
                    "lag": lag,
                    "horizon": horizon,
                    "nrmse": nrmse
                })
                print(f"{model_name} nrmse: {nrmse}")
        except FileNotFoundError:
            print(f"Dataset for lag={lag}, horizon={horizon} not found.")
            continue

results_df = pd.DataFrame(results)
best_models = results_df.sort_values("nrmse").head(5)
print("\nBest 5 Models:")
best_models

plt.figure(figsize=(12, 6))

for model_name in ["Ridge", "Lasso", "ElasticNet"]:
    # Filter results for the current model and lag 10
    model_results_lag10 = results_df[(results_df["model"] == model_name) & (results_df["lag"] == 10)]
    plt.plot(model_results_lag10["horizon"], model_results_lag10["nrmse"], marker='o', label=model_name)

plt.xlabel("Forecast Horizon")
plt.ylabel("NRMSE")
plt.title("NRMSE of Ridge, Lasso, and ElasticNet (Lag=10)")
plt.xticks(ridge_results_lag10["horizon"])  # Use the horizons from ridge results for xticks
plt.legend()
plt.grid(True)

