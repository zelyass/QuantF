# Structured Products: Heston and Vasicek

Design and analysis of two capital-protected structured notes, written for **FIN8645 – Produits structurés financiers** (UQAM, Winter 2026, Prof. Alexandre F. Roch) by Zakaria El Yassini (Concordia University).

- **Product 1 (equity):** coupons that increase with the realized volatility of the underlying (ZCN, BMO S&P/TSX Capped Composite Index ETF).
- **Product 2 (rates):** coupons that decrease with the short rate.
- **Models:** Heston for the equity underlying, Vasicek for interest rates, both calibrated to market data.
- **Valuation:** under the risk-neutral measure Q, with the initial-equity condition V0 = N0.
- **Risk analysis:** Monte Carlo under the physical measure P, with VaR95% and ES95% read as shortfall measures, and a first-order stochastic dominance comparison between configurations.

The full report (in French) is in [`report/Projet_produits_structures.pdf`](report/Projet_produits_structures.pdf).

## Repository layout

```
report/   Final report (PDF, French)
code/
  heston/ MATLAB live scripts: implied dividend yield q under Q, Heston calibration, Product 1 and 2 equity components
  rates/  MATLAB live script: Vasicek calibration to the CAD OIS curve
data/     CORRA series (Bank of Canada) and calibrated Vasicek parameters
```

Suggested order: `Part I Calibration of q under Q` → `Calibration Heston parametres` → `Vasicek_Whole` → `Product 1` / `Product 2`.

## Requirements

MATLAB (R2026a) with the Financial Instruments Toolbox (`optByHestonNI`).

## Data

The market data used in the report comes from Bloomberg and is **not redistributed** here. See [`data/README.md`](data/README.md) for what is included and how to re-create the rest.

## Tools disclosure

Grammarly was used for language corrections. ChatGPT was used occasionally for formatting questions (LaTeX, bibliography, figure layout).

## Licence

Code: MIT (see `LICENSE`). The report is © 2026 Zakaria El Yassini, all rights reserved.
