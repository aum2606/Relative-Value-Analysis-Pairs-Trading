# Relative Value Analysis & Pairs Trading

This project implements a comprehensive framework for relative value analysis and pairs trading strategy, leveraging statistical methods to identify trading opportunities in financial markets. The strategy is based on identifying pairs of securities that are cointegrated and trading them based on mean-reversion principles.

## Project Overview

Pairs trading is a market-neutral strategy that seeks to profit from temporary mispricings between related securities. The key steps involved are:

1. **Correlation Analysis**: Identify pairs of securities with high historical correlation.
2. **Cointegration Testing**: Filter for pairs that exhibit a stationary spread (cointegration).
3. **Signal Generation**: Create trading signals based on deviations from the historical relationship.
4. **Backtesting**: Evaluate the strategy's performance using historical data.
5. **Performance Analysis**: Assess the strategy's risk-adjusted returns and other key metrics.

## Project Structure

```
relative_value_pairs_trading/
│
├── data/
│   ├── raw/                    # Raw data storage
│   │   └── Portfolio_prices.csv  # Input portfolio price data
│   └── processed/              # Processed data files
│       └── price_data.csv        # Processed price data
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_pairs_selection.ipynb
│   ├── 03_cointegration_tests.ipynb
│   ├── 04_trading_signals.ipynb
│   └── 05_backtesting.ipynb
│
├── src/                        # Source code
│   ├── data/                   # Data handling modules
│   │   ├── data_ingestion.py     # Data loading functionality
│   │   └── data_preprocessing.py # Data cleaning and preparation
│   ├── utils/                  # Utility functions
│   │   ├── logger.py             # Logging functionality
│   │   └── exception.py          # Custom exceptions
│   ├── analysis/               # Analysis modules
│   │   ├── correlation_analysis.py  # Correlation calculation
│   │   └── cointegration_tests.py   # Cointegration testing
│   │
│   ├── strategies/             # Trading strategies
│   │   ├── pairs_trading.py       # Pairs trading implementation
│   │   └── mean_reversion.py      # Mean reversion methods
│   │
│   └── backtesting/            # Backtesting framework
│       ├── backtest_engine.py     # Backtesting engine
│       ├── performance_metrics.py # Performance calculation
│       └── visualization.py       # Results visualization
│
├── reports/                    # Generated reports and visualizations
│   ├── pairs_trading_strategy_report.txt
│   └── backtest_results.csv
│
├── logs/                       # Log files
├── README.md                   # Project documentation
├── requirements.txt            # Project dependencies
└── main.py                     # Main execution script
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/aum2606/relative_value_pairs_trading.git
   cd relative_value_pairs_trading
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Place your portfolio price data in the `data/raw/` directory with the name `Portfolio_prices.csv`.

## Usage

### Running the Full Pipeline

To run the full analysis pipeline:

```bash
python main.py
```

### Customizing Parameters

The main script accepts several command-line arguments to customize the strategy:

```bash
python main.py --lookback 60 --entry 2.0 --exit 0.5 --stop_loss 3.0 --max_holding 20 --position_size 0.5 --min_correlation 0.7 --top_n_pairs 10 --test_method engle_granger
```

Parameters:
- `--lookback`: Lookback period for calculating spread statistics (default: 60)
- `--entry`: Z-score threshold to enter a position (default: 2.0)
- `--exit`: Z-score threshold to exit a position (default: 0.5)
- `--stop_loss`: Maximum z-score deviation before triggering a stop loss (default: 3.0)
- `--max_holding`: Maximum number of periods to hold a position (default: 20)
- `--position_size`: Size of position to take (1.0 = 100% of available capital) (default: 0.5)
- `--min_correlation`: Minimum correlation threshold for pair selection (default: 0.7)
- `--top_n_pairs`: Number of top pairs to select for testing (default: 10)
- `--test_method`: Cointegration test method ('adf', 'engle_granger', or 'johansen') (default: 'engle_granger')

### Running Individual Components

Each component can also be run separately using the corresponding Python scripts in the `src` directory. For example:

```python
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessing

# Data ingestion
data_ingestion = DataIngestion()
raw_df = data_ingestion.read_data()

# Data preprocessing
data_preprocessor = DataPreprocessing()
processed_df = data_preprocessor.preprocess_data(raw_df)
```

## Input Data Format

The input CSV file should have the following columns:
- `Date`: Date of the price record (format: YYYY-MM-DD)
- `Ticker`: Stock ticker symbol
- `Open`, `High`, `Low`, `Close`: Price data
- `Adjusted`: Adjusted closing price
- `Returns`: Daily returns (optional, can be calculated)
- `Volume`: Trading volume

## Methods and Formulas

### Correlation Analysis

**Pearson Correlation Coefficient**
$$\rho_{x,y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

**Spearman Rank Correlation**
$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

### Cointegration Tests

**Augmented Dickey-Fuller (ADF) Test** for stationarity of the price spread between two stocks.

**Engle-Granger Two-Step Cointegration Test**:
1. Estimate the cointegration regression: $y_t = \alpha + \beta x_t + \epsilon_t$
2. Apply the ADF test to the residuals $\epsilon_t$ to check for stationarity.

**Johansen Cointegration Test** for multivariate cointegration analysis.

### Trading Signal Generation

**Z-Score**
$$z_t = \frac{y_t - \mu_y}{\sigma_y}$$

**Bollinger Bands**
- Upper Band: $\mu_y + k \sigma_y$
- Lower Band: $\mu_y - k \sigma_y$

**Half-Life of Mean Reversion**
$$HL = \frac{\ln(2)}{\lambda}$$

### Performance Metrics

**Sharpe Ratio**
$$SR = \frac{R_p - R_f}{\sigma_p}$$

**Maximum Drawdown**
$$MDD = \max_{t \in (0,T)} \left( \max_{s \in (0,t)} V_s - V_t \right)$$

## Workflow Steps

1. **Data Preprocessing**:
   - Load raw price data from CSV
   - Convert date formats
   - Handle missing values
   - Calculate returns if not present

2. **Correlation Analysis**:
   - Calculate the correlation matrix of all tickers
   - Identify ticker pairs with correlation above the threshold
   - Sort pairs by correlation strength

3. **Cointegration Testing**:
   - For each highly correlated pair, test for cointegration
   - Use ADF, Engle-Granger, or Johansen test methods
   - Filter pairs that show statistically significant cointegration

4. **Pairs Selection**:
   - Select the top cointegrated pairs for trading
   - Calculate hedge ratios for each pair

5. **Trading Signal Generation**:
   - Calculate the spread between each pair
   - Normalize the spread using z-score
   - Generate entry and exit signals based on z-score thresholds

6. **Backtesting**:
   - Simulate trading based on the generated signals
   - Track portfolio value, position sizing, and transaction costs
   - Apply position limits and risk management rules

7. **Performance Evaluation**:
   - Calculate key performance metrics (Sharpe ratio, drawdown, etc.)
   - Visualize equity curve and drawdowns
   - Analyze trade statistics (win rate, average win/loss)

## Results Interpretation

After running the pipeline, the following results will be generated in the `reports` directory:

1. **Portfolio Value Chart**: Shows the equity curve of the strategy over time.
2. **Drawdown Analysis**: Visualizes the drawdowns experienced by the strategy.
3. **Returns Distribution**: Shows the distribution of returns.
4. **Performance Metrics**: Key metrics including Sharpe ratio, maximum drawdown, win rate, etc.
5. **Pair Analysis Charts**: Visualizes the price series, spread, and z-score for each pair.

## Implementation Notes

- The implementation uses a modular approach where each component is isolated
- The pipeline follows the single responsibility principle
- Error handling is implemented using a custom exception class
- Detailed logging helps track the execution flow
- Visualization tools help interpret results visually

## Dependencies

This project requires the following Python packages:

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.12.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

## License

[GNU License](LICENSE)

## Contributors

- Aum Parmar aumparmar@gmail.com

## Acknowledgments

This project is inspired by academic research on pairs trading and statistical arbitrage strategies.

## References

- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. The Review of Financial Studies, 19(3), 797-827.
- Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: representation, estimation, and testing. Econometrica: journal of the Econometric Society, 251-276.
- Vidyamurthy, G. (2004). Pairs Trading: Quantitative methods and analysis. John Wiley & Sons.