# FINDATA

## A Python package for downloading, cleaning, and modeling stock market data


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is a package that aims to make it easier to obtain up-to-date stock market and economic data, derive and engineer features based on said data, and test out modelling strategies in a flexible manner.

What began as a absolute rat's nest of scripts and Jupyter notebooks is now a modular, class-based, mostly-clean Python package. (The refactoring took many weekends, some rather painful, but has been totally worth it.)

Current functionality includes:
* Downloading from
    * Yahoo Finance (daily stock data, options chains)
    * Quandl
    * Investing.com
* Engineering features like
    * Measures of trend (e.g. moving averages, Z-score)
    * Measures of volatility (e.g. standard deviation, average true range)
* Building and backtesting models
    * LightGBM
    * ElasticNet

### Built With

* [Selenium](https://www.selenium.dev/)
* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/)



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* [Python 3](https://www.python.org/downloads/)

If downloading data other than from Yahoo Finance or Quandl:

* [Firefox](https://www.mozilla.org/en-CA/firefox/new/)
* [geckodriver.exe](https://github.com/mozilla/geckodriver/releases/)

### Installation

1. Clone this repo:
   ```sh
   git clone https://github.com/aleysia/findata.git
   ```
2. Run `setup.py` to install the `findata` package and dependencies:
   ```sh
   python setup.py install
   ```
3. Create a `config.yml` file containing the path to `geckodriver.exe` as follows:
   ```yaml
   geckodriver:
       path: YOUR/PATH/TO/geckodriver.exe
   ```

### Optional steps
1. You can scrape economic data from [Investing.com](https://www.investing.com). The following indicators are implemented:
    * [MOVE](https://www.investing.com/indices/ice-bofaml-move-historical-data)
    * [Manufacturing PMI](https://www.investing.com/economic-calendar/manufacturing-pmi-829)

   Sign up for an account at , and place the following lines in `config.yml`:
   ```yaml
    investing:
        email: YOUR@LOGIN.EMAIL
        pwd: YOUR_PASSWORD
   ```
2. You can import data from Quandl by signing up for an account, and adding the following to `config.yml`:
    ```yaml
    quandl:
        api-key: YOUR_API_KEY
    ```

    By default, AAII sentiment and BTC/USD are downloaded.

<!-- USAGE EXAMPLES -->
## Usage

Below is example code that
1. Downloads a handful of stock tickers (listed in `/data/symbols/example.csv`), including SPY.
2. Derives features such as lagged differences, moving average, average true range, etc.
    * Feel free to implement additional features!
3. Builds models using LightGBM and default hyperparameters to predict 5-day change in SPY closing price.
    * This is a sort of backtest (of modelling, not trading strategy) using a sliding window
4. Prints out the last 20 rows of out-of-sample performance, and forward-looking prediction.

```python
# Data download
import findata.DataImporter as di

importer = di.DataImporter()
importer.get_ticker_list(name='example')
importer.import_tickers()

# Feature engineering
import findata.FeatureEngineer as fe

engineer = fe.FeatureEngineer()
engineer.set_params(index_etf='SPY', start_date='2020-01-01')
engineer.combine_sources()
data = engineer.data

# Prediction using LightGBM
import findata.ModelTrainer as mt

trainer = mt.LightGBMModelTrainer(data, y_variable='SPY_close', date_col=None, target_horizon_days=5)
trainer.setup_data(sliding_window=5)
trainer.set_params(loss_function='rmse')
trainer.run(skip_tuning=True)

print(trainer.training_performance.tail(20))
print(trainer.forward_prediction)
```

If you want all S&P500 symbols, the following will scrape the list from Wikipedia. If you run `import_tickers()` at this point, please be aware that it's going to take a while due to rate-limiting.

```python
import findata.DataImporter as di

importer = di.DataImporter()
importer.get_ticker_list(name='sp500')
```



<!-- ROADMAP -->
## Roadmap

In no particular order, things I'd eventually like to implement:

* Documentation
* Informative error messages for common mistakes
* Test scripts
* Add more indicators and engineered features
* Visualizations (feature importance, diagnostics, etc.)


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Best README Template](https://github.com/othneildrew/Best-README-Template)
