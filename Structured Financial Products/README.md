# Structured Products: Heston and Vasicek

Design and analysis of two capital-protected structured notes, written for **FIN8645 – Produits structurés financiers** (UQAM, Winter 2026, Prof. Alexandre F. Roch) by Zakaria El Yassini (Concordia University).

- **Product 1 (equity):** coupons that increase with the realized volatility of the underlying (ZCN, BMO S&P/TSX Capped Composite Index ETF).
- **Product 2 (rates):** coupons that decrease with the short rate.
- **Models:** Heston for the equity underlying, Vasicek for interest rates, both calibrated to market data.
- **Valuation:** under the risk-neutral measure Q, with the initial-equity condition V0 = N0.
- **Risk analysis:** Monte Carlo under the physical measure P, with VaR95% and ES95% read as shortfall measures, and a first-order stochastic dominance comparison between configurations.

The full report (in French) is in under Reports and is in French

## Requirements

MATLAB (R2026a) with the Financial Instruments Toolbox 

## Data

The market data used in the report comes from Bloomberg and Bank of Canada. I will unfortunately not share them directly here for Terms reasons.

## Tools disclosure

Grammarly was used for language corrections. ChatGPT was used occasionally for formatting questions (LaTeX, bibliography, figure layout).

## Licence

Code: MIT. The report is © 2026 Zakaria El Yassini, all rights reserved.
