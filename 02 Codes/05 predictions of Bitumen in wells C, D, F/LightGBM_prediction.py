import pickle
import numpy as np



# load data
data = np.genfromtxt('./Bitumen/Data/trn_tst/Prediction/tst_dat.dat')



# Load the XGBoost model
with open('./Bitumen/Results/LightGBM/lgbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Make predictions using the loaded XGBoost model
lgbm_y_pred_test = model.predict(data)

# Save the predictions to a file that MATLAB can read
np.savetxt('./Bitumen/Results/LightGBM/lgbm_y_pred_test.dat', lgbm_y_pred_test)
