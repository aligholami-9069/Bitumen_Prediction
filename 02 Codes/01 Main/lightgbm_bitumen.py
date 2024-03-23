import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import pickle


# load data
X_trn = np.genfromtxt('./Bitumen/Data/trn_tst/trn_dat.dat')
X_tst = np.genfromtxt('./Bitumen/Data/trn_tst/tst_dat.dat')
y_trn = np.genfromtxt('./Bitumen/Data/trn_tst/trn_lbl.dat')
y_tst = np.genfromtxt('./Bitumen/Data/trn_tst/tst_lbl.dat')

# Create LightGBM Regressor
lgb_regressor = LGBMRegressor(boosting_type='gbdt',
                              objective='regression',
                              num_leaves=30,
                              n_estimators=100,
                              min_child_samples=1,
                              reg_alpha=0.6,
                              reg_lambda=0.7,
                              learning_rate=0.9,
                              feature_fraction=0.3,
                              bagging_fraction=0.5,
                              bagging_freq=1)

# Train the model
model = lgb_regressor.fit(X_trn, y_trn, eval_set=[(X_tst, y_tst)])
                  
# Save the model object using pickle
with open('C:/Users/Ali/Desktop/Bitumen/Results/LightGBM/lgbm_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Make predictions on test data
y_pred = model.predict(X_tst)



# save the prediction
np.savetxt('C:/Users/Ali/Desktop/Bitumen/Results/LightGBM/lgbm_y_pred.dat', np.insert(y_pred, 0, 0))


lgbm_mse = mean_squared_error(y_tst, y_pred)
lgbm_r = np.corrcoef(y_tst, y_pred)[0][1]