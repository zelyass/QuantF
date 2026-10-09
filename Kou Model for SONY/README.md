# Bayesian Estimation of the Kou Jump-Diffusion Model: SONY

Equity returns drift, compress, and occasionally jump. The moves are discontinuous and asymmetric, and their tails are much heavier than a normal law allows. The Kou (2002) jump-diffusion model was built for this setting. This project fits it from the ground up to Sony (`SONY`) log-returns from 2020 to 2026.

The implementation is self-contained. We build a custom Gibbs sampler and let the data and the prior settle the estimate between them.

---

## Why This Model

Three features of financial return series are difficult for standard diffusion models to capture:

- Tails are heavy. Crashes and rallies occur far more often than a Gaussian predicts.
- Returns are asymmetric. Large losses tend to be larger than large gains.
- Prices jump. Discontinuous moves appear that a continuous-path model can only approximate.

The Kou double-exponential model handles all three. From the data it estimates whether jumps occurred, how large they were, which direction they tended to take, and how often they arise. Each answer comes with its posterior uncertainty.

---

## The Model

Log-returns split into a diffusive part and a jump part:
```
r_k = μΔ + σ√Δ · ε_k + J_k · Y_k

  ε_k ~ N(0, 1)
  J_k ~ Bernoulli(π)
```

On a jump day, the jump size follows a double-exponential law:
```
       ⎧  Exp(η₁)   with probability p        (upward jump)
Y_k ~ ⎨
       ⎩ -Exp(η₂)   with probability (1 − p)  (downward jump)
```

The asymmetry between `η₁` and `η₂` lets the model reflect the common observation that equity crashes are sharper and faster than rallies.

---

## Bayesian Estimation

### Priors from the data

Priors are anchored to the sample before inference begins. Each one reflects what the observed series already shows:

- `μ` and `σ` are anchored to sample moments.
- Jump candidates are flagged by a volatility threshold rule.
- Tail decay rates are estimated by excess-over-threshold. This isolates the exponential decay using only observations in the tail, so the estimate reflects the tail alone.

For the jump probability, the prior is tight:
```
π ~ Beta(2, 200)
```

It encodes a belief of roughly a 1% daily jump probability, consistent with decades of equity market evidence. It also restrains the model's tendency to explain excess variance through spurious jumps.

### Gibbs sampling

Posterior inference runs through a hand-built Gibbs sampler. Each iteration carries two latent variables per observation:

- `J_k`, whether a jump occurred on day *k*
- `Y_k`, the size of that jump, if one occurred

Both update alongside the model parameters through conjugate full conditionals. Computations run on the log scale wherever underflow is a risk.

---

## Details That Matter

Several design choices make inference stable and well identified.

### Keeping jumps honest

Small diffusive fluctuations would otherwise leak into the jump component and inflate `π`. A minimum jump size prevents this:
```
|Y_k| ≥ y_min = 2σ
```

This boundary keeps each component focused on its own job: the diffusion explains small moves, and the jumps explain large ones.

### Estimating the tails from the tails

Raw sample averages for `η₁` and `η₂` mix the bulk of the distribution with its tail. The excess-over-threshold approach concentrates estimation on the tail:
```
E[Y − u | Y > u]
```

This yields a consistent estimator of the exponential decay rate. It draws only on observations above the threshold `u`, which are the observations that define tail behavior.

### The continuous-time limit

The Bernoulli approximation behind the discrete sampler holds when:
```
λΔ ≪ 1
```

We check this condition explicitly. When it holds, the discrete procedure tracks the continuous-time process it approximates. If it fails, the mismatch would go unnoticed.

---

## Outputs

### Exploratory diagnostics
- Price and return series with annotated jump events
- QQ plots against the fitted model

### MCMC health checks
- Trace plots and autocorrelation functions
- Effective sample size (ESS) and R-hat for all parameters

### Posterior results
- Full marginal distributions for `μ`, `σ`, `π`, `η₁`, `η₂`, `p`
- Derived quantities: annualized volatility, jump intensity `λ`, and expected jump magnitudes by direction

### Predictive validation
- Simulated vs. observed return distributions
- Kurtosis comparison to check the tail fit

---

## Generated Files

| File | Description |
|------|-------------|
| `sony_empirical_priors.pkl` | Serialized prior hyperparameters |
| `sony_mcmc_results_2025.pkl` | Full MCMC chain output |
| `sony_posterior_summary_2025.csv` | Parameter posterior statistics |
| `*.png` | All diagnostic plots |

---

## Stack

| Library | Role |
|---------|------|
| `NumPy` | Sampling and array operations |
| `SciPy` | Truncated distributions, optimization |
| `Pandas` | Data ingestion and return computation |
| `Matplotlib` | Diagnostics and posterior visualizations |

---

## Takeaways

Fitting a jump-diffusion model is routine. Fitting it well requires a deliberate choice at each stage. The Kou model is flexible enough to match the asymmetric tails of equity returns, and that flexibility calls for careful prior design and identifiability constraints.

The choices made here (informative priors, a minimum jump size, threshold-based tail estimation) determine what the model is allowed to explain. Those choices decide whether the posterior reflects the data or the model's appetite for complexity.

- The Kou model captures asymmetric tail behavior in equity returns.
- Bayesian estimation gives uncertainty quantification for every parameter.
- Jump-diffusion models tend to overfit jump intensity when the prior is loosely specified.
- Excess-over-threshold estimation of tail parameters is strongly preferred to naive moment matching.
