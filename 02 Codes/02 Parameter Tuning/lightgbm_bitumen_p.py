import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import csv


csv_path = './Bitumen/Results/LightGBM/Param_OPT/lgbm_param_tuning.csv'

# load data
X_trn = np.genfromtxt('./Bitumen/Data/trn_tst/trn_dat.dat')
X_tst = np.genfromtxt('./Bitumen/Data/trn_tst/tst_dat.dat')
y_trn = np.genfromtxt('./Bitumen/Data/trn_tst/trn_lbl.dat')
y_tst = np.genfromtxt('./Bitumen/Data/trn_tst/tst_lbl.dat')

i = 0

with open(csv_path, 'w', newline='') as file:
    wr = csv.writer(file)

    for num_leaves in range(30,31,1):
        for n_estimators in range(100,101,1):
           for min_child_samples in range(1,2,1):
              for bagging_freq in range(1,3,1):
                  for reg_alpha in np.arange(0.5,1,0.1):
                     for reg_lambda in np.arange(0.6,1,0.1):
                         for learning_rate in np.arange(0.5,1,0.1):
                             for feature_fraction in np.arange(0.2,1,0.1):
                                  for bagging_fraction in np.arange(0.4,1,0.1):
                                      
                                      i = i + 1
                                      print(i)
                                      lgb_regressor = LGBMRegressor(boosting_type='gbdt',
                                                                    objective='regression',num_leaves=num_leaves,
                                                                   n_estimators=n_estimators, min_child_samples=min_child_samples,
                                                                   bagging_freq=bagging_freq, reg_alpha=reg_alpha,
                                                                   reg_lambda=reg_lambda, learning_rate=learning_rate,
                                                                   feature_fraction=feature_fraction, bagging_fraction=bagging_fraction)
                                      
                                      model = lgb_regressor.fit(X_trn, y_trn, eval_set=[(X_tst, y_tst)])
                                      
                                      y_pred = model.predict(X_tst)
                                      
                                      lgbm_mse = mean_squared_error(y_tst, y_pred)
                                      lgbm_r = np.corrcoef(y_tst, y_pred)[0][1]
                                      
                                      params = [i, num_leaves, n_estimators, min_child_samples, bagging_freq, reg_alpha, reg_lambda, learning_rate, feature_fraction, bagging_fraction, lgbm_mse, lgbm_r]
                                      wr.writerow(params)
                  file.close()


