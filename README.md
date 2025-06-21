
<a id="readme-top"></a>

<div align="center">
  <h3 align="center">Portfolio Optimization using Markowitz Model</h3>
  <p align="center">
    A complete implementation of the Markowitz Mean-Variance Optimization framework in Python. This project selects a diversified asset set, optimizes asset allocation, backtests performance, and visualizes risk-return tradeoffs for a portfolio of INR 1,00,000.
    <br />
    <a href="https://github.com/yourusername/markowitz-portfolio-optimizer"><strong>Explore the code »</strong></a>
    <br />
    <br />
    <a href="#usage">View Usage</a>
    ·
    <a href="https://github.com/yourusername/markowitz-portfolio-optimizer/issues/new?template=bug_report.md&labels=bug">Report Bug</a>
    ·
    <a href="https://github.com/yourusername/markowitz-portfolio-optimizer/issues/new?template=feature_request.md&labels=enhancement">Request Feature</a>
  </p>
</div>

---
## Developer Contacts

**Bhavya Bharti** – [GitHub](https://github.com/bhavyabhart)

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

---

## About The Project

This project implements the **Markowitz Mean-Variance Optimization Model**, a foundational approach in modern portfolio theory. It seeks to construct a well-diversified investment portfolio by minimizing portfolio risk for a target level of expected return.

The analysis is based on:
- **Training period:** April 2019 – March 2022  
- **Backtesting period:** April 2022 – March 2025  
- **Initial Investment:** ₹1,00,000  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Built With

- Python
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- CVXPY  
- yfinance

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites

```bash
pip install yfinance pandas numpy matplotlib seaborn cvxpy
```

### Installation

```bash
git clone https://github.com/yourusername/markowitz-portfolio-optimizer.git
cd markowitz-portfolio-optimizer
```

```bash
jupyter notebook Portfolio_Optimization_Markowitz.ipynb
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Usage

- Modify the tickers list
- Run notebook to:
  - Download data
  - Optimize portfolio
  - Backtest performance
  - View charts and metrics

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Documentation

### Asset Selection

Assets were selected across sectors (e.g. equity, ETF, pharma, gold, bonds) based on historical volatility and low correlation to enhance diversification.

### Optimized Portfolio

- Method: Markowitz Mean-Variance Optimization  
- Constraints: No short selling (0 ≤ w ≤ 0.33), full investment (∑w = 1)  
- Objective: Minimize risk for target return

### Portfolio PnL

Backtested from April 2022 – March 2025 with ₹1,00,000 capital using optimized weights.

### Performance Metrics

- Total Return  
- Annualized Return  
- Annualized Volatility  
- Sharpe Ratio  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contributing

1. Fork the repository  
2. Create a new branch (`git checkout -b feature/AmazingFeature`)  
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)  
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contact

Project Link: [https://github.com/yourusername/markowitz-portfolio-optimizer](https://github.com/yourusername/markowitz-portfolio-optimizer)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
