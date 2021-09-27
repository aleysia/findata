import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import itertools
import datetime as dt
from datetime import timedelta

from multiprocessing import Pool
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import cophenet
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

import findata.utils as fdutils


class FeatureEngineer():


    def __init__(self, read_dir='data/raw', write_dir='data/processed'):
        self.read_dir = os.path.join(read_dir, '')
        self.write_dir = os.path.join(write_dir, '')
        Path(self.write_dir).mkdir(parents=True, exist_ok=True)
        self.logger = fdutils.new_logger('FeatureEngineer')


    def set_params(self,
                   start_date='2011-04-11', end_date=None,
                   keep_duplicate='last',
                   filters={'avg_20d_pricevol':1000000, 'n_records':120},
                   price_lags=[20], 
                   change_depth=2, change_lags=[5, 10, 20],
                   price_features=['atr', 'zscore', 'pricevol', 'vol', 'vwap', 'demark',
                                   'dreturn', 'chaikin', 'corr', 'std'],
                   drop_cols=['price', 'open', 'high', 'low', 'close', 'dividends', 
                              'symbol', 'volume', 'vol.', 'adj close', 'change %'],
                   index_etf='SPY', price_variables=['close', 'open'], profit_variable='price'):
        self.start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = dt.datetime.now() if end_date is None else dt.datetime.strptime(end_date, '%Y-%m-%d')
        if self.start_date.month==self.end_date.month and \
                self.start_date.day==self.end_date.day and self.start_date.year==self.end_date.year:
            raise ValueError('Start and end date are the same!')
        self.keep_duplicate = keep_duplicate
        self.filters = filters
        self.price_features = price_features
        self.price_lags = price_lags
        self.change_depth = change_depth
        self.change_lags = change_lags
        self.drop_cols = drop_cols
        self.index_etf = index_etf
        self.price_variables = price_variables if isinstance(price_variables, list) else [price_variables]
        self.profit_variable = profit_variable


    def passes_filters(self, dff):
        PASS = True
        for filter in self.filters.keys():
            try:
                if 'avg_20d_pricevol' in self.filters.keys():
                    val = (dff['close']*dff['volume']).rolling(20).mean()
                    PASS = PASS & (val.values[-1] >= self.filters['avg_20d_pricevol'])
                if 'n_records' in self.filters.keys():
                    PASS = PASS & (len(dff) >= self.filters['n_records'])
            except Exception as e:
                self.logger.debug(f'Was not able to apply filter {filter}:\n' + str(e))
        return PASS


    def format_cols(self, dff):
        dff.columns = [col.lower() for col in dff.columns]
        dff['date'] = pd.to_datetime(dff['date'])
        dff = dff.set_index('date')
        dff = dff.loc[~dff.index.duplicated(keep=self.keep_duplicate)]
        return dff


    def clean_cols(self, dff, f, keep=[]):
        dff = self.drop_columns(dff)
        dff.columns = [os.path.split(f)[1].replace('.csv','') + '_' + col for col in dff.columns]
        return dff


    def drop_columns(self, dff, keep=[]):
        return dff.drop(columns=list(set(self.drop_cols) - set(keep)), errors='ignore')

    
    def filter_dates(self, dff):
        dff = dff.loc[(dff.index>=self.start_date) & (dff.index<=self.end_date)]
        # Need to filter out dates with lots of NaN indicators (e.g. holiday in US but not in Canada -- 
        # nothing's going to happen those days anyway), without filtering out a bunch of dates just because
        # a couple ETFs didn't exist yet or something
        perc_nas = dff.isnull().mean(axis=1)
        dff = dff.loc[perc_nas<0.90]

        return dff


    def create_price_features(self, dff, index_dff=None, price_features=None):
        price_features = price_features or self.price_features

        try:
            for price_variable in self.price_variables:
                if price_variable not in dff.columns:
                    continue

                util_cols = pd.DataFrame(index=dff.index)
                util_cols['ma20d'] = dff[price_variable].rolling(20).mean()
                util_cols['high52wk'] = dff['high'].rolling(250).max()
                util_cols['low52wk'] = dff['low'].rolling(250).min()
                util_cols['high20d'] = dff['high'].rolling(20).max()
                util_cols['low20d'] = dff['low'].rolling(20).min()
                util_cols['prev_close'] = dff['close'].shift(1)
                util_cols['dreturn'] = dff[price_variable] / dff[price_variable].shift(1) - 1
                util_cols['A'] = dff['high'] - util_cols['prev_close']
                util_cols['B'] = util_cols['prev_close'] - dff['low']
                util_cols['C'] = dff['high'] - dff['low']

                dff['dreturn'] = dff[price_variable] / dff[price_variable].shift(1) - 1

                for feat_name, lag in itertools.product(price_features, self.price_lags):
                    # Lagged indicators
                    if feat_name == 'dreturn':
                        dff[feat_name + str(lag)] = dff[price_variable] / dff[price_variable].shift(lag) - 1
                    if feat_name == 'zscore':
                        dff[feat_name + str(lag)] = (dff[price_variable] - dff[price_variable].rolling(lag, lag//2).mean()) \
                            / dff[price_variable].rolling(lag, lag//2).std()
                    if feat_name == 'std':
                        dff[feat_name + str(lag)] = dff[price_variable].rolling(lag, lag//2).std()
                    if feat_name == 'pricevol':
                        dff[feat_name + str(lag)] = (dff['close']*dff['volume']).rolling(lag, lag//2).mean()
                    if feat_name == 'vol':
                        dff[feat_name + str(lag)] = dff['volume'].rolling(lag, lag//2).mean() / dff['volume'][0]
                    if feat_name == 'atr':
                        dff[feat_name] = util_cols[['A','B','C']].max(axis=1)
                    if feat_name == 'vwap':
                        dff[feat_name + str(lag)] = ((dff['close'] - dff['close'].shift(1)) * dff['volume']).rolling(lag, lag//2).sum()
                    if feat_name == 'demark':
                        demax = pd.Series([x if x>0 else 0 for x in (dff['high'] - dff['high'].shift(1))])
                        demin = pd.Series([x if x>0 else 0 for x in (dff['low'].shift(1) - dff['low'])])
                        dff['demark'] = (demax.rolling(lag, lag//2).mean() \
                            / (demax.rolling(lag, lag//2).mean() + demin.rolling(lag, lag//2).mean())).values

                    # Immediate indicators
                    if feat_name == 'chaikin':
                        dff[feat_name] = dff['volume'] * ((dff['close'] - dff['low']) - (dff['high'] - dff['close'])) \
                            / util_cols['C']

                    if index_dff is None:
                        continue

                    # Comparison-to-index indicators
                    if 'dreturn' not in index_dff.columns:
                        index_dff['dreturn'] = index_dff[price_variable] / index_dff[price_variable].shift(1) - 1
                    if feat_name == 'corr':
                        tmp = util_cols[['dreturn']].merge(index_dff[['dreturn']], left_index=True, right_index=True)
                        tmp['corr'] = tmp['dreturn_x'].rolling(lag, lag//2).corr(tmp['dreturn_y'])
                        dff = dff.merge(tmp[['corr']], how='left', left_index=True, right_index=True)
        except Exception as e:
            self.logger.debug(f'Exception while creating price-based feature \'{price_variable}\':\n' + str(e))
        finally:
            return dff


    def read_file(self, f):
        try:
            df = pd.read_csv(f, index_col=None)
            df = self.format_cols(df)
            return df
        except Exception as e:
            self.logger.debug(f'Exception encountered while reading file {f}:\n' + str(e))
            return None


    def read_all_tickers(self, files, separate_index=False, keep_names=False):
        ticker_dfs = []
        index_df = None
        for f in files:
            df = self.read_file(f)
            if df is None:
                continue

            ticker_name = os.path.split(f)[1].replace('.csv','')
            if keep_names:
                df['symbol'] = ticker_name

            if separate_index and ticker_name == self.index_etf:
                index_df = df
            elif df is not None and self.passes_filters(df):
                ticker_dfs.append(df)
        return ticker_dfs, index_df


    def drop_constant(self, df):
        return df.loc[:, (df != df.iloc[-1]).any()]


    def combine_sources(self):
        self.logger.info('Processing raw files')

        dfs = []
        for f in glob.iglob(self.read_dir + '*.csv', recursive=True):
            df = self.read_file(f)
            df = self.filter_dates(df)
            if df is not None:
                df = self.clean_cols(df, f, keep='price')
                df = self.create_change_features(df)
                dfs.append(df)

        for f in glob.iglob(self.read_dir + 'sectors/*.csv', recursive=True):
            df = self.read_file(f)
            df = self.filter_dates(df)
            if df is not None:
                df = self.clean_cols(df, f)
                df = self.create_change_features(df)
                dfs.append(df)
        
        ticker_files = glob.iglob(self.read_dir + 'tickers/*.csv', recursive=True)
        ticker_dfs, index_df = self.read_all_tickers(ticker_files, separate_index=True, keep_names=True)
        ticker_dfs2 = []
        for df in ticker_dfs:
            df = self.create_price_features(df, index_df)
            df = self.filter_dates(df)
            df = self.drop_columns(df)
            ticker_dfs2.append(df)

        index_df = self.create_price_features(index_df)
        index_df = self.drop_columns(index_df, keep=['close'])
        index_df.columns = [self.index_etf + '_' + col for col in index_df.columns]
        dfs.append(index_df)

        ticker_columns = ticker_dfs2[0].columns
        ticker_avgs = pd.DataFrame()
        for col in ticker_columns:
            tmp = pd.concat([dff[col] for dff in ticker_dfs2], axis=1)
            ticker_avgs[col] = tmp.mean(axis=1)
        ticker_avgs = self.create_change_features(ticker_avgs)
        dfs.append(ticker_avgs)

        self.data = pd.concat(dfs, axis=1)
        self.data = self.filter_dates(self.data)
        self.data = self.data.sort_index().replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
        self.logger.info('Finished processing files!')


    def create_change_features(self, df, exclude=[]):
        lagged_dfs = [df]
        for lag in self.change_lags:
            df_lag = df.select_dtypes(include='number').drop(columns=exclude, errors='ignore').copy()
            for _ in range(self.change_depth):
                df_lag = df_lag.diff(lag)
                df_lag.columns = [col + '_' + str(lag) for col in df_lag.columns]
                lagged_dfs.append(df_lag)
        return pd.concat(lagged_dfs, axis=1)


    def create_interactions(self, df):
        return df


    def create_agg_features(self, df):
        return df


    def calculate_opex(self, dates):
        """
        Return datetime.date for monthly option expiration given year and month
        """
        def third_saturday(year, month):
            # The 15th is the lowest third day in the month
            third = dt.date(int(year), int(month), 15)
            # What day of the week is the 15th?
            w = third.weekday()
            # Saturday is weekday 5
            if w != 5:
                # Replace just the day (of month)
                third = third.replace(day=(15 + (5 - w) % 7))
            return third

        # dates = pd.date_range(self.start_date, self.end_date)
        months = np.unique(pd.DatetimeIndex(dates).strftime('%Y-%m').tolist())
        opex = pd.Series([third_saturday(*m.split('-')) - timedelta(days=1) for m in months])
        return opex


    def create_time_features(self, df):
        opex = self.calculate_opex(df.reset_index()['date'])
        df['OpEx'] = df.reset_index()['date'].isin(opex.values).values
        df['OpEx'] = df['OpEx'].fillna(False)
        df['Days2opex'] = (df.groupby(df['OpEx'].cumsum()).cumcount(ascending=False)+1).astype(float)
        df.drop(columns='OpEx', inplace=True)

        df['DayOfWeek'] = pd.to_datetime(df.reset_index()['date']).dt.dayofweek.astype(float).values
        df['DayOfMonth'] = pd.to_datetime(df.reset_index()['date']).dt.day.astype(float).values
        df['Week'] = pd.to_datetime(df.reset_index()['date']).dt.isocalendar().week.astype(float).values
        return df


    def create_all_features(self, df):
        df = self.create_time_features(df)
        return df

