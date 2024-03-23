import numpy as np
import xgboost
from sklearn.metrics import mean_squared_error
import csv

csv_path = './Results/XGBoost/Param_OPT/xgb_param_tuning.csv'

# load data
X_trn = np.genfromtxt('./Bitumen/Data/trn_tst/trn_dat.dat')
X_tst = np.genfromtxt('./Bitumen/Data/trn_tst/tst_dat.dat')
y_trn = np.genfromtxt('./Bitumen/Data/trn_tst/trn_lbl.dat')
y_tst = np.genfromtxt('./Bitumen/Data/trn_tst/tst_lbl.dat')


i = 0

with open(csv_path, 'w', newline='') as file:
    wr = csv.writer(file)

    for n_estimators in range(100,3000,100):
        for max_depth in range(1,21,2):
            for eta in np.arange(0,1,0.1):
                for subsample in np.arange(0,1,0.1):
                    for colsample_bytree in np.arange(0,1,0.1):
                        
                        i = i + 1
                        print(i)
                        model = xgboost.XGBRegressor(n_estimators=n_estimators,
                                                     max_depth=max_depth, eta=eta,
                                                     subsample=subsample, colsample_bytree=colsample_bytree)
                        model.fit(X_trn, y_trn)
                        y_pred = model.predict(X_tst)
                        
                        xgb_mse = mean_squared_error(y_tst, y_pred)
                        xgb_r = np.corrcoef(y_tst, y_pred)[0][1]
                        
                        params = [i, n_estimators, max_depth, eta, subsample, colsample_bytree, xgb_mse, xgb_r]
                        wr.writerow(params)
    file.close()




