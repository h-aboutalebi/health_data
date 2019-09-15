import lightgbm as lgb


class My_lightgbm():

    def fit(self, x_train, y_train):
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        lgb_train = lgb.Dataset(x_train, y_train)
        self.gbm = lgb.train(params, lgb_train,
                             num_boost_round=20)

    def predict(self,X_test):
        return self.gbm.predict(X_test, num_iteration=self.gbm.best_iteration)

