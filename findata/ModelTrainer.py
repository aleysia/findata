import pandas as pd
import numpy as np
import datetime as dt
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import spearmanr, SpearmanRConstantInputWarning
from sklearn.model_selection import train_test_split
import warnings
from sklearn.linear_model import ElasticNet, ElasticNetCV

import findata.utils as fdutils

optuna.logging.set_verbosity(optuna.logging.ERROR)

warnings.simplefilter('ignore', SpearmanRConstantInputWarning)


class ModelTrainer:


    def __init__(self, X, y_variable='close', relative=True, target_horizon_days=10, clip=None,
                 id_cols=None, date_col='date', start_date=None, ignore_cols=['Date', 'date']):
        self.target_horizon_days = target_horizon_days
        self.date_col = date_col
        if date_col not in X.columns:
            X = X.reset_index()
        X[date_col] = pd.to_datetime(X[date_col])
        if start_date is not None:
            X = X.loc[X[date_col]>=dt.datetime.strptime(start_date, '%Y-%m-%d')]
        self.X = X.drop(columns=[y_variable])
        if target_horizon_days is not None:
            self.Y_org = 100*(X[y_variable].shift(-target_horizon_days)/X[y_variable] - 1) if relative \
                else X[y_variable].shift(-target_horizon_days) - X[y_variable]
        else:
            self.Y_org = X[y_variable]
        self.Y = self.Y_org.clip(clip[0], clip[1]) if clip is not None else self.Y_org
        self.type = 'classification' if self.Y.isin([0,1]).all() else 'regression'
        self.logger = fdutils.new_logger('ModelTrainer')
        self.ignore_cols = ignore_cols
        self.id_cols = None if id_cols is None else (id_cols if isinstance(id_cols, list) else [id_cols])


    def setup_data(self, size_train=240, size_eval=None, size_test=None, sliding_window=1,
                   washout_period=None, by_date=False):
        self.sliding_window = sliding_window or self.target_horizon_days or 10
        size_test = size_test or self.sliding_window
        size_eval = size_eval or size_train // 4

        self.size_train = size_train
        self.size_eval = size_eval
        self.size_test = size_test

        washout_period = washout_period or self.target_horizon_days or sliding_window

        self.by_date = by_date
        if not by_date:
            if self.target_horizon_days is not None:
                self.X_test = self.X.iloc[-self.target_horizon_days:]
                self.X = self.X.iloc[:-self.target_horizon_days]
                self.Y = self.Y.iloc[:-self.target_horizon_days]
                self.Y_org = self.Y_org.iloc[:-self.target_horizon_days]
            else:
                self.X_test = self.X.iloc[-self.size_test:]
                self.X = self.X.iloc[:-self.size_test]
                self.Y = self.Y.iloc[:-self.size_test]
                self.Y_org = self.Y_org.iloc[:-self.size_test]

            max_trainable = len(self.X)
            self.index_train = list(range(0, max_trainable - size_train,
                                        self.sliding_window))
            self.index_eval = list(range(size_train + washout_period,
                                        max_trainable - size_eval,
                                        self.sliding_window))
            self.index_test = list(range(size_train + washout_period + size_eval + washout_period,
                                        max_trainable - size_test,
                                        self.sliding_window))
        else:
            dates = pd.Series(self.X[self.date_col].unique()).sort_values()
            if self.target_horizon_days is not None:
                self.X_test = self.X.loc[self.X[self.date_col].isin(dates.iloc[-self.target_horizon_days:])]
                self.Y = self.Y.loc[self.X[self.date_col].isin(dates.iloc[:-self.target_horizon_days])]
                self.Y_org = self.Y_org.loc[self.X[self.date_col].isin(dates.iloc[:-self.target_horizon_days])]
                self.X = self.X.loc[self.X[self.date_col].isin(dates.iloc[:-self.target_horizon_days])]
            else:
                self.X_test = self.X.loc[self.X[self.date_col].isin(dates.iloc[-1:])]
                self.Y = self.Y.loc[self.X[self.date_col].isin(dates.iloc[:-1]) & (~self.Y.isna())]
                self.Y_org = self.Y_org.loc[self.X[self.date_col].isin(dates.iloc[:-1]) & (~self.Y.isna())]
                self.X = self.X.loc[self.X[self.date_col].isin(dates.iloc[:-1]) & (~self.Y.isna())]

            max_trainable = len(self.X[self.date_col].unique())
            self.index_train = list(range(0, max_trainable - size_train,
                                        self.sliding_window))
            self.index_eval = list(range(size_train + washout_period,
                                        max_trainable - size_eval,
                                        self.sliding_window))
            self.index_test = list(range(size_train + washout_period + size_eval + washout_period,
                                        max_trainable - size_test,
                                        self.sliding_window))
            self.dates = dates

        # If any of the train/eval run over test then remove them
        while len(self.index_train) > len(self.index_test):
            self.index_train.pop()
        while len(self.index_eval) > len(self.index_test):
            self.index_eval.pop()


    def set_params(self, filter=None, filter_thres=0.1):
        self.params = {'filter': filter, 'filter_thres': filter_thres}


    def feature_selection(self, train_X, train_Y):
        features = self.X.drop(columns=self.ignore_cols, errors='ignore').columns
        if self.params['filter'] == 'auc':
            vals = [roc_auc_score(train_X[col], train_Y) for col in features]
            vals = [v if v>0.5 else 1-v for v in vals]
        elif self.params['filter'] == 'spearmanr':
            vals = [spearmanr(train_X[col], train_Y)[0] for col in features]
            vals = [v if v>0 else -v for v in vals]
        else:
            return features
        features = [features[i] for i in range(len(features)) if vals[i]>=self.params['filter_thres']]
        return features


    def train(self, train_X_feat, train_Y, eval_X_feat, eval_Y, skip_tuning=False):
        return None


    def predict(self, model, test_X_feat):
        return np.zeroes(len(test_X_feat))


    def run(self, skip_tuning=False):
        self.logger.info(f'Beginning training! Total training rows available: {len(self.X)}')
        train_perf = pd.DataFrame()
        train_pred = pd.DataFrame()
        for i in range(len(self.index_train)):
            self.logger.info(f'Round {i}/{len(self.index_train)-1}: train {self.index_train[i]}-{self.index_train[i]+self.size_train-1}, ' + 
                             f'eval {self.index_eval[i]}-{self.index_eval[i]+self.size_eval-1}, ' + 
                             f'test {self.index_test[i]}-{self.index_test[i]+self.size_test-1}')
            LAST_ITER = (i+1) == len(self.index_train)

            if not self.by_date:
                train_X = self.X.iloc[self.index_train[i] : self.index_train[i]+self.size_train]
                train_Y = self.Y.iloc[self.index_train[i] : self.index_train[i]+self.size_train]
                eval_X = self.X.iloc[self.index_eval[i] : self.index_eval[i]+self.size_eval]
                eval_Y = self.Y.iloc[self.index_eval[i] : self.index_eval[i]+self.size_eval]

                if not LAST_ITER:
                    test_X = self.X.iloc[self.index_test[i] : self.index_test[i]+self.size_test]
                    test_Y = self.Y.iloc[self.index_test[i] : self.index_test[i]+self.size_test]
                    org_Y = self.Y_org.iloc[self.index_test[i] : self.index_test[i]+self.size_test]
                else:
                    test_X = self.X.iloc[self.index_test[i] : ]
                    test_Y = self.Y.iloc[self.index_test[i] : ]
                    org_Y = self.Y_org.iloc[self.index_test[i] : ]
            else:
                train_start = self.dates[self.index_train[i]]
                train_end = self.dates[self.index_train[i]+self.size_train]
                eval_start = self.dates[self.index_eval[i]]
                eval_end = self.dates[self.index_eval[i]+self.size_eval]
                test_start = self.dates[self.index_test[i]]
                test_end = self.dates[self.index_test[i]+self.size_test]

                train_X = self.X.loc[self.X[self.date_col].isin(pd.date_range(train_start, train_end))]
                train_Y = self.Y.loc[self.X[self.date_col].isin(pd.date_range(train_start, train_end))]
                eval_X = self.X.loc[self.X[self.date_col].isin(pd.date_range(eval_start, eval_end))]
                eval_Y = self.Y.loc[self.X[self.date_col].isin(pd.date_range(eval_start, eval_end))]

                if not LAST_ITER:
                    test_X = self.X.loc[self.X[self.date_col].isin(pd.date_range(test_start, test_end))]
                    test_Y = self.Y.loc[self.X[self.date_col].isin(pd.date_range(test_start, test_end))]
                    org_Y = self.Y_org.loc[self.X[self.date_col].isin(pd.date_range(test_start, test_end))]
                else:
                    test_X = self.X.loc[self.X[self.date_col].isin(pd.date_range(test_start, self.dates.max()))]
                    test_Y = self.Y.loc[self.X[self.date_col].isin(pd.date_range(test_start, self.dates.max()))]
                    org_Y = self.Y_org.loc[self.X[self.date_col].isin(pd.date_range(test_start, self.dates.max()))]
     
            self.debug_Xtrain_ = train_X
            self.debug_ytrain_ = train_Y
            self.debug_Xtest_ = test_X
            self.debug_ytest_ = test_Y

            features = self.feature_selection(train_X, train_Y)
            model = self.train(train_X[features], train_Y, eval_X[features], eval_Y, skip_tuning)
            y_hat = self.predict(model, test_X[features])

            rho, _ = spearmanr(y_hat, test_Y)
            train_perf = train_perf.append({'round': i, 
                                            'start_date': test_X[self.date_col].values[0],
                                            'end_date': test_X[self.date_col].values[-1],
                                            'spearmanr': rho if not np.isnan(rho) else 0,
                                            'rmse': mean_squared_error(test_Y, y_hat),
                                            'accuracy': ((test_Y>0)==(y_hat>0)).mean()}, ignore_index=True)
            tmp = pd.DataFrame(data={'date': test_X[self.date_col],
                                     'yhat': y_hat,
                                     'ytrain': test_Y,
                                     'y': org_Y})
            if self.id_cols is not None:
                for col in self.id_cols:
                    tmp[col] = test_X[col].values
            train_pred = train_pred.append(tmp, ignore_index=True)

        self.model = model
        y_hat = self.predict(model, self.X_test[features])
        forward_pred = pd.DataFrame(data={'date': self.X_test[self.date_col], 'y_hat': y_hat})
        if self.id_cols is not None:
            for col in self.id_cols:
                forward_pred[col] = self.X_test[col]

        self.model = model
        self.training_performance = train_perf
        self.training_predictions = train_pred
        self.forward_prediction = forward_pred.iloc[-1]



class LightGBMModelTrainer(ModelTrainer):


    def __init__(self, X, y_variable='close', relative=True, target_horizon_days=10, clip=[-5, 5],
                 id_cols=None, date_col='date', start_date=None, ignore_cols=['Date', 'date']):
        super().__init__(X, y_variable, relative, target_horizon_days, clip, id_cols, date_col, start_date, ignore_cols)
        self.logger = fdutils.new_logger('LightGBMModelTrainer')


    def set_params(self, filter=None, filter_thres=0.1, trials=10,
                   num_threads=5, early_stopping_rounds=100,
                   max_iter=1000, loss_function='rmse'):
        self.params = locals()


    def custom_asymmetric_train(self, y_pred, data):
        y_true = data.get_label()
        residual = (y_true - y_pred).astype("float")
        grad = np.where(residual<0, -2*2.0*residual, -2*residual)
        hess = np.where(residual<0, 2*2.0, 2.0)
        return grad, hess


    def custom_asymmetric_valid(self, y_pred, data):
        y_true = data.get_label()
        residual = (y_true - y_pred).astype("float")
        loss = np.where(residual < 0, (residual**2)*2.0, residual**2) 
        return "custom_asymmetric_eval", np.max(loss), False


    def train_evaluate(self, params, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1234)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        model = lgb.train(params, train_data, valid_sets=[valid_data], verbose_eval=False)
            # fobj=self.custom_asymmetric_train, feval=self.custom_asymmetric_valid)
        scores = model.best_score['valid_0']
        score = scores[list(scores.keys())[0]]
        return score


    def tune(self, lgb_params, X, y):
        def objective(trial):
            opt_params = {'num_leaves': trial.suggest_int('num_leaves', 2, 50),
                          'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
                          'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
                          'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                          'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
                          'lambda_l1': trial.suggest_loguniform('lambda_l1', 0.1, 10),
                          'lambda_l2': trial.suggest_loguniform('lambda_l2', 0.1, 10),
                          'max_bin': trial.suggest_int('max_bin', 2, 100)}
            opt_params['max_depth'] = trial.suggest_int('max_depth', 1, np.log2(opt_params['num_leaves']) // 1)
            return self.train_evaluate({**lgb_params, **opt_params}, X, y)

        study = optuna.create_study(direction='minimize')
        if self.previous_best_params is not None:
            study.enqueue_trial(self.previous_best_params)
        study.optimize(objective, n_trials=self.params['trials'])
        self.previous_best_params = study.best_params
        return study.best_params


    def train(self, train_X_feat, train_Y, eval_X_feat, eval_Y, skip_tuning=False):
        train_data = lgb.Dataset(train_X_feat, train_Y)
        eval_data = lgb.Dataset(eval_X_feat, eval_Y)

        lgb_params = {'objective': 'binary_logloss' if self.type == 'classification' else self.params['loss_function'],
                #'min_sum_hessian_in_leaf': 0.00001, 'min_data_in_leaf': 1,
                'verbosity': -1, 'num_threads': self.params['num_threads']}
        lgb_params['metric'] = lgb_params['objective']
        if not skip_tuning:
            self.previous_best_params = None
            best_params = self.tune(lgb_params, train_X_feat, train_Y)
            lgb_params = {**lgb_params, **best_params}

        model = lgb.train(lgb_params, train_data, num_boost_round=self.params['max_iter'],
                          valid_sets=[eval_data], verbose_eval=False,
                          #fobj=self.custom_asymmetric_train, feval=self.custom_asymmetric_valid,
                          early_stopping_rounds=self.params['early_stopping_rounds'])
        return model


    def predict(self, model, test_X_feat):
        return model.predict(test_X_feat)



class ElasticNetModelTrainer(ModelTrainer):


    def __init__(self, X, y_variable='close', relative=True, target_horizon_days=10, clip=[-5, 5],
                 id_cols=None, date_col='date', start_date=None, ignore_cols=['Date', 'date']):
        X = X.loc[~np.isnan(X[y_variable])]
        X = X.fillna(0)
        super().__init__(X, y_variable, relative, target_horizon_days, clip, id_cols, date_col, start_date, ignore_cols)
        self.logger = fdutils.new_logger('ElasticNetModelTrainer')


    def set_params(self, filter=None, filter_thres=0.1, cv=5, max_iter=1000, n_jobs=5,
                  n_alphas=100, l1_ratio=[0.01, 0.05, .1, .5, .7, .9, .95, .99, 1]):
        self.params = locals()


    def train(self, train_X_feat, train_Y, eval_X_feat, eval_Y, skip_tuning=False):
        if not skip_tuning:
            model = ElasticNetCV(n_alphas=self.params['n_alphas'], l1_ratio=self.params['l1_ratio'])
        else:
            model = ElasticNet(alpha=0.5)

        model.fit(train_X_feat.append(eval_X_feat), train_Y.append(eval_Y))

        return model


    def predict(self, model, test_X_feat):
        return model.predict(test_X_feat)
