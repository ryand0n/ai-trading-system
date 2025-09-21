# AI Trading System

## Overview

This project implements an AI-based trading system that can learn and test trading strategies on stock market data. The system supports both a manual rule-based strategy and a machine learning-based strategy learner. It reads historical price data, computes technical indicators as features, and uses these features to generate trading signals. The strategies are then evaluated on both in-sample and out-of-sample data to assess their performance.

## Features

- **Reads in historical price data** for a specified stock symbol and date range.
- **Calculates technical indicators** (such as SMA, Bollinger Bands, momentum, etc.) to use as features for trading decisions.
- **Manual Strategy**: Uses hand-crafted rules based on indicators to generate buy/sell signals.
- **Strategy Learner**: Uses machine learning (e.g., Q-Learning) to learn an optimal trading policy from the data.
- **Simulates trades** and computes portfolio values, taking into account market impact and commission costs.
- **Compares performance** of the learned strategy against a benchmark.

## How to Use

### Requirements

- Python 3.x
- Required packages: `numpy`, `pandas`, `matplotlib`

### Running the Project

You can run the main script to test either the manual or strategy learner:

```bash
python main.py --learner [manual|strategy] [options]
```

#### Command Line Arguments

**Required:**
- `--learner`: Choose strategy type (`manual` for rule-based, `strategy` for ML-based)

**Optional (with defaults):**
- `--symbol`: Stock symbol to trade (default: JPM)
- `--start`: Training start date YYYY-MM-DD (default: 2008-01-01)
- `--end`: Training end date YYYY-MM-DD (default: 2009-12-31)
- `--sv`: Starting portfolio value (default: 100000)
- `--impact`: Market impact factor (default: 0.005)
- `--commission`: Commission per trade (default: 9.95)
- `--future_start`: Out-of-sample start date (default: 2010-01-01)
- `--future_end`: Out-of-sample end date (default: 2011-12-31)

#### Example Usage

```bash
# Test ML strategy on AAPL with custom parameters
python main.py --learner strategy --symbol AAPL --start 2020-01-01 --end 2020-12-31 --sv 100000

# Test manual strategy with different commission
python main.py --learner manual --symbol MSFT --commission 5.00 --impact 0.01
```
