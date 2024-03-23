# Bitumen_Prediction
# Estimation of solid bitumen content in hydrocarbon reservoirs: fusion of individual machine learning models and petrophysical well-logging data in a committee machine 

### By:
### Ali Gholami Vijouyeh <sup>a</sup>, Maha Raoof Hamoudi <sup>b</sup>, Dyana Aziz Bayz <sup>c</sup>, Ali Kadkhodaie <sup>a</sup>

<sup>a</sup> Earth Sciences Department, Faculty of Natural Science, University of Tabriz, Tabriz, Iran

<sup>b</sup> Department of Natural Resources Engineering and Management, University of Kurdistan Hewler, Erbil, Iraq

<sup>c</sup> Department of Petroleum Engineering, College of Engineering, Knowledge University, Erbil, Iraq

---


## Content


 ### __01 Main__

* #### __correlation_plot.m__
    Calculates and shows the correlation between different fields of data using Matlab R2022b.

* #### __create_data.m__
    Creates training and test data from selected features of main data in Matlab R2022b. The results are saved in:
".\01 Data\test_data\\" folder.

* #### __fuzzy_logic.m__
    Implements TS-FIS algorithm in Matlab R2022b. The results are saved in:
".\03 Output files\ 03 Stand-alone algorithms\Fuzzy Logic\\" folder.

* #### __lightgbm_bitumen.m__
    Loads and calls the results of LightGBM algorithm modelling in Matlab R2022b, implemented in Python 3.11. The results are saved in:
".\03 Output files\ 03 Stand-alone algorithms\LightGBM\\" folder.

* #### __lightgbm_bitumen.py__
    Implements LightGBM algorithm in Python 3.11. The results are saved in:
 ".\03 Output files\ 03 Stand-alone algorithms\LightGBM\\" folder.

* #### __neural_network.m__
    Implements Neural Network algorithm in Matlab R2022b. The script reads the proper network from ".\03 Output files \03 Stand-alone algorithms\Neural Network\Param_OPT \608_net.mat\". The results are saved in:
".\03 Output files \03 Stand-alone algorithms\Neural Network " folder.

* #### __neuro_fuzzy.m__
    Implements Neuro-Fuzzy algorithm in Matlab R2022b. It reads the network from ".\03 Output files \03 Stand-alone algorithms\Neuro Fuzzy\Net \NF_bitumen.mat\". The results are saved in:
 ".\03 Output files \03 Stand-alone algorithms\Neuro Fuzzy\" folder.

* #### __optimization.m__
    This file implements all optimization committee machine algorithms in Matlab R2022b, which are "GA", "SA", and "ACOR". The results are saved in:
 ".\03 Output files\04 Optimization by committee machines\\" folder.

* #### __RBF.m__
    Implements Radial Basis Function algorithm in Matlab R2022b. The results are stored in:
 ".\03 Output files\ 03 Stand-alone algorithms\RBF\\" folder.

* #### __xgboost.m__
    Loads and calls the results of XGBoost algorithm modelling in Matlab R2022b, implemented in Python 3.11. The results are stored in:
 ".\03 Output files\ 03 Stand-alone algorithms\XGBoost\\" folder.

* #### __XGBoost.py__
    Implements XGBoost algorithm in Python 3.11. The outcomes are stored in:
 ".\03 Output files\ 03 Stand-alone algorithms\XGBoost\\" folder.
------


### __02 Parameter Tuning__

* #### __lightgbm_bitumen_p.py__
    This file has been used to parameter tuning of LightGBM algorithm using Python 3.11. The results are stored in:
 ".\03 Output files\03 Stand-alone algorithms\LightGBM\Param_OPT\\" folder.

* #### __neural_network_parameters.m__
    This file has been used to parameter tuning for back-propagation neural network algorithm using Matlab R2022b. The results are stored in:
 ".\03 Output files\03 Stand-alone algorithms\Neural Network\Param_OPT \\" folder. 
During this operation, the most optimal model was concluded to be: 608_net.mat (".\03 Output files\03 Stand-alone algorithms\Neural Network\Param_OPT \608_net.mat \").

* #### __optimization_01_ga.m__
    This file has been used to parameter tuning for GA optimization algorithm in Matlab R2022b. The results are stored in:
 ".\03 Output files\04 Optimization by committee machines\GA\\" folder. 

* #### __optimization_02_sa.m__
    This file has been used to parameter tuning for SA optimization algorithm in Matlab R2022b. The results are stored in:
 ".\03 Output files\04 Optimization by committee machines\SA\\" folder. 

* #### __optimization_03_aco.m__
    This script tunes parameter for ACOR optimization algorithm in Matlab R2022b. The results are stored in:
".\03 Output files\04 Optimization by committee machines\ACO\\" folder. 

* #### __RBF_param.m__
This file has been used to parameter tuning for Radial Basis Function algorithm using Matlab R2022b. The results are stored in:
 ".\03 Output files\03 Stand-alone algorithms\RBF\Param_OPT\\" folder. 

* #### __XGBoost_p.py__
This file has been used to parameter tuning of XGBoost algorithm using Python 3.11. The results are stored in:
 ".\03 Output files\03 Stand-alone algorithms\XGBoost\Param_OPT \\" folder.
 
---

### __03 DT and CGR prediction__

#### CGR prediction

* #### __create_data.m__
Creates training and test data from selected features of main data in Matlab R2022b to predict CGR well-logging data. 

* #### __neural_network.m__
Implements Neural Network algorithm in Matlab R2022b to estimate CGR well-logging data. The script reads the proper network from ".\03 Output files\02 CGR and DT prediction\CGR Results\Neural Network \Param_OPT \353_net.mat\". The results are saved in:
".\03 Output files\02 CGR and DT prediction\CGR Results\Neural Network\" folder.

* #### __neural_network_parameters.m__
This file has been used to parameter tuning for back-propagation neural network algorithm using Matlab R2022b to estimate CGR values. The outcomes are stored in:
 ".\03 Output files\02 CGR and DT prediction\CGR Results\Neural Network \Param_OPT \\" folder. 
In this operation, the most optimal model was concluded to be: 353_net.mat (".\03 Output files\02 CGR and DT prediction\CGR Results\Neural Network\Param_OPT \353_net.mat\").

* #### __script_plot_3d.m__
This script was used to figure the models obtained from parameter tuning of CGR prediction (in Matlab R2022b). The outcomes are stored in:
".\03 Output files\02 CGR and DT prediction\CGR Results\Neural Network\Param_OPT\ann_mse_3d_plot.png & ann_r_3d_plot.png\".

* #### __tst_prediction.m__
This script estimates and draws the CGR values using the well-logging input data from the model obtained by BP-NN in Matlab R2022b. The outcomes of each well are stored in:
".\03 Output files\02 CGR and DT prediction\CGR Results\Prediction of well b (CGR)\ ". 
".\03 Output files\02 CGR and DT prediction\CGR Results\Prediction of well d (CGR)\ ". 

---

#### DT prediction

* #### __create_data.m__
Creates training and test data from selected features of main data in Matlab R2022b to predict DT well-logging data. 

* #### __neural_network.m__
Implements Neural Network algorithm in Matlab R2022b to estimate DT well-logging data. The script reads the proper network from ".\03 Output files\02 CGR and DT prediction\ DT Results\Neural Network \Param_OPT \217_net.mat\". The results are saved in:
".\03 Output files\02 CGR and DT prediction\ DT Results\Neural Network\" folder.

* #### __neural_network_parameters.m__
This file has been used to parameter tuning for back-propagation neural network algorithm using Matlab R2022b to estimate DT values. The outcomes are stored in:
 ".\03 Output files\02 CGR and DT prediction\DT Results\Neural Network \Param_OPT \\" folder. 
In this operation, the most optimal model was concluded to be: 217_net.mat (".\03 Output files\02 CGR and DT prediction\DT Results\Neural Network\Param_OPT \217_net.mat\").

* #### __script_plot_3d.m__
This script was used to figure the models obtained from parameter tuning of DT prediction (in Matlab R2022b). The outcomes are stored in:
".\03 Output files\02 CGR and DT prediction\DT Results\Neural Network\Param_OPT\ann_mse_3d_plot.png & ann_r_3d_plot.png\".

* #### __tst_prediction.m__
This script estimates and draws the DT values using the well-logging input data from the model obtained by BP-NN in Matlab R2022b. The outcomes of each well are stored in:
".\03 Output files\02 CGR and DT prediction\DT Results\Prediction of Ahwaz_307 (DT) \ ". 
".\03 Output files\02 CGR and DT prediction\ DT Results\Prediction of SD_3 (DT) \ ". 

---

### __04 Multi-variable linear regression__

* #### __three_regression.m__
This file implements multi-variable linear regression method in Matlab R2022b. The results are saved in:
".\03 Output files\05 Multi-variable linear regression\".
05 predictions of Bitumen in wells C, D, F
These scripts are designed for prediction of bitumen in other wells using the models obtained from AI systems in this study. To run it:
#### In first step:
Run tst_create_data.m script, which loads, creates and divides the well-logging input data. 

#### In second step:
Run LightGBM_prediction.py, which loads the model constructed with LightGBM algorithm from Python 3.11 to Matlab R2022b. The results are stored in:
".\03 Output files\03 Stand-alone algorithms\LightGBM\lgbm_y_pred_test.dat\".

#### In third step: 
Run XGBoost_prediction.py, which loads the model constructed with XGBoost algorithm from Python 3.11 to Matlab R2022b. The results are stored in:
".\03 Output files\03 Stand-alone algorithms\XGBoost\xgb_y_pred_test.dat\".

#### In final step: 
Run tst_prediction.m, which load, model and predict the targets using MVLR (as the best algorithm resulted in this study). The results are saved in:
".\03 Output files\07 all wells predictions\".



