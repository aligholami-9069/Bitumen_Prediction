import numpy as np
import xgboost
from sklearn.metrics import mean_squared_error
import pickle



# load data
X_trn = np.genfromtxt('./Bitumen/Data/trn_tst/trn_dat.dat')
X_tst = np.genfromtxt('./Bitumen/Data/trn_tst/tst_dat.dat')
y_trn = np.genfromtxt('./Bitumen/Data/trn_tst/trn_lbl.dat')
y_tst = np.genfromtxt('./Bitumen/Data/trn_tst/tst_lbl.dat')


# define model
model = xgboost.XGBRegressor(n_estimators=200, max_depth=7, eta=0.7, subsample=0.7, colsample_bytree=0.4)

# fit model
model.fit(X_trn, y_trn)

# Save the model object using pickle
with open('C:/Users/Ali/Desktop/Bitumen/Results/XGBoost/xgb_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# make a prediction
y_pred = model.predict(X_tst)

# save the prediction
np.savetxt('C:/Users/Ali/Desktop/Bitumen/Results/XGBoost/xgb_y_pred.dat', np.insert(y_pred, 0, 0))


xgb_mse = mean_squared_error(y_tst, y_pred)
xgb_r = np.corrcoef(y_tst, y_pred)[0][1]

