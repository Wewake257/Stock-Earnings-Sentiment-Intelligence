# STOCK EARNINGS SURPRISE AND SENTIMENT INTELLIGENCE REPORT

**Generated:** September 28, 2026 at 12:02
**Author:** Vivek Kumar | Data Analytics Portfolio
**Coverage:** 48 Tickers | 6 Sectors | 2,000 Events

---
## Executive Summary

| KPI | Value |
|-----|-------|
| Total Events | 2,000 |
| Overall Beat Rate | 42.1% |
| Avg Sentiment | -0.004 |
| Best Sector EQS | Consumer |
| Weakest Sector EQS | Energy |
| Best ML Model | Logistic Regression AUC=0.9883 |
| Strong Buy Hit Rate | 100.0% |
| Strong Sell Hit Rate | 100.0% |

---
## Earnings Surprise Analysis
| Surprise Class | Count | % | Avg Day-of Reaction |
|---|---|---|---|
| Strong Beat | 273 | 13.7% | +26.87% |
| Moderate Beat | 570 | 28.5% | +10.60% |
| In-Line | 367 | 18.4% | -0.36% |
| Moderate Miss | 558 | 27.9% | -10.56% |
| Strong Miss | 232 | 11.6% | -26.53% |

---
## Sector Scorecard
| Sector | Beat Rate | Avg EPS Surprise | Avg Reaction | Avg EQS | Guidance Raise |
|---|---|---|---|---|---|
| Consumer | 48% | +1.3% | +2.10% | -0.013 | 29% |
| Finance | 45% | +0.6% | +0.87% | -0.042 | 31% |
| Healthcare | 39% | +0.2% | +0.05% | -0.095 | 28% |
| Industrials | 39% | +0.2% | +0.17% | -0.114 | 26% |
| Technology | 40% | +0.2% | -0.06% | -0.128 | 27% |
| Energy | 42% | +0.5% | +0.52% | -0.131 | 24% |

---
## Composite Signal Performance
| Signal | Events | Avg EQS | Avg Sentiment | Day-of Reaction | Hit Rate |
|---|---|---|---|---|---|
| Strong Buy | 214 | +1.242 | +0.865 | +27.62% | 100.0% |
| Buy | 554 | +0.307 | +0.678 | +11.83% | 97.1% |
| Hold | 440 | -0.135 | +0.075 | +1.85% | 56.1% |
| Sell | 532 | -0.416 | -0.727 | -10.76% | 2.6% |
| Strong Sell | 260 | -1.272 | -0.826 | -24.44% | 0.0% |

---
## ML Model Tournament
| Model | CV AUC | Test AUC |
|---|---|---|
| Logistic Regression | 0.9905 | 0.9883 |
| Random Forest | 0.9883 | 0.9821 |
| Gradient Boosting | 0.9876 | 0.9859 |

**Champion:** Logistic Regression AUC=0.9883 | **Top Feature:** EPS Surprise Z-Score

---
## Architecture

- EarningsDataEngine: Multi-stock realistic earnings simulation
- FinancialSentimentEngine: Loughran-McDonald NLP with VADER-style weighting
- EarningsSurpriseAnalytics: Z-score normalization, EQS, composite signal
- MLEngine: Model tournament (LR / RF / GBM) with 5-fold CV
- VizEngine: 7 dark-theme professional financial charts

---
## Portfolio CV Relevance

| Skill Demonstrated | Industry Signal |
|---|---|
| Financial NLP / Sentiment Analysis | Quant Research, FinTech Analytics |
| Earnings Surprise Quantification | Investment Banking, Equity Research |
| ML Signal Generation | Quantitative Finance, Hedge Funds |
| Sector-Level Portfolio Analytics | Asset Management |
| PEAD Anomaly Quantification | Academic / Research Finance |

---

*Vivek Kumar - Stock Earnings Sentiment Intelligence | Data Analytics Portfolio*