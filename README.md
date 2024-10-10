# Bitumen_Prediction
# Estimation of solid bitumen content in hydrocarbon reservoirs: Fusion of individual machine learning models and petrophysical well-logging data in a two-step committee machine

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
    Creates training and test data from selected features of main data in Matlab R2022b. 

* #### __fuzzy_logic.m__
    Implements TS-FIS algorithm in Matlab R2022b. 

* #### __lightgbm_bitumen.m__
    Loads and calls the results of LightGBM algorithm modelling in Matlab R2022b, implemented in Python 3.11. 

* #### __lightgbm_bitumen.py__
    Implements LightGBM algorithm in Python 3.11. 

* #### __neural_network.m__
    Implements Neural Network algorithm in Matlab R2022b. 

* #### __neuro_fuzzy.m__
    Implements Neuro-Fuzzy algorithm in Matlab R2022b. 

* #### __optimization.m__
    This file implements all optimization committee machine algorithms in Matlab R2022b, which are "GA", "SA", "ACOR", "CMA-ES" and "GWO". 

* #### __RBF.m__
    Implements Radial Basis Function algorithm in Matlab R2022b. 

* #### __xgboost.m__
    Loads and calls the results of XGBoost algorithm modelling in Matlab R2022b, implemented in Python 3.11. 

* #### __XGBoost.py__
    Implements XGBoost algorithm in Python 3.11. 
------


### __02 Parameter Tuning__

* #### __lightgbm_bitumen_p.py__
    This file has been used to parameter tuning of LightGBM algorithm using Python 3.11. 

* #### __neural_network_parameters.m__
    This file has been used to parameter tuning for back-propagation neural network algorithm using Matlab R2022b. 

* #### __optimization_01_ga.m__
    This file has been used to parameter tuning for GA optimization algorithm in Matlab R2022b/R2017b. 

* #### __optimization_02_sa.m__
    This file has been used to parameter tuning for SA optimization algorithm in Matlab R2022b/R2017b. 

* #### __optimization_03_aco.m__
    This script tunes parameter for ACOR optimization algorithm in Matlab R2022b/R2017b. 

* #### __optimization_05_gwo.m__
    This script tunes parameter for GWO optimization algorithm in Matlab R2022b/R2017b.
  
* #### __optimisation_with_cmaes_tuning.m__
    This script tunes parameter for CMA-ES optimization algorithm in Matlab R2022b.
    
* #### __RBF_param.m__
This file has been used to parameter tuning for Radial Basis Function algorithm using Matlab R2022b. 

* #### __XGBoost_p.py__
This file has been used to parameter tuning of XGBoost algorithm using Python 3.11. 

---

### __03 DT and CGR prediction__

#### CGR prediction

* #### __create_data.m__
Creates training and test data from selected features of main data in Matlab R2022b to predict CGR well-logging data. 

* #### __neural_network.m__
Implements Neural Network algorithm in Matlab R2022b to estimate CGR well-logging data. 

* #### __neural_network_parameters.m__
This file has been used to parameter tuning for back-propagation neural network algorithm using Matlab R2022b to estimate CGR values. 

* #### __script_plot_3d.m__
This script was used to figure the models obtained from parameter tuning of CGR prediction (in Matlab R2022b). 

* #### __tst_prediction.m__
This script estimates and draws the CGR values using the well-logging input data from the model obtained by BP-NN in Matlab R2022b. 

---

#### DT prediction

* #### __create_data.m__
Creates training and test data from selected features of main data in Matlab R2022b to predict DT well-logging data. 

* #### __neural_network.m__
Implements Neural Network algorithm in Matlab R2022b to estimate DT well-logging data. 

* #### __neural_network_parameters.m__
This file has been used to parameter tuning for back-propagation neural network algorithm using Matlab R2022b to estimate DT values. 

* #### __script_plot_3d.m__
This script was used to figure the models obtained from parameter tuning of DT prediction (in Matlab R2022b). 

* #### __tst_prediction.m__
This script estimates and draws the DT values using the well-logging input data from the model obtained by BP-NN in Matlab R2022b. 

---

### __04 Multi-variable linear regression__

* #### __three_regression.m__
This file implements multi-variable linear regression method in Matlab R2022b/R2017b. 

---

### __05 predictions of Bitumen in wells C, D, F__

These scripts are designed for prediction of bitumen in other wells using the models obtained from AI systems in this study. To run it:
#### In first step:
Run tst_create_data.m script, which loads, creates and divides the well-logging input data. 

#### In second step:
Run LightGBM_prediction.py, which loads the model constructed with LightGBM algorithm from Python 3.11 to Matlab R2022b. 

#### In third step: 
Run XGBoost_prediction.py, which loads the model constructed with XGBoost algorithm from Python 3.11 to Matlab R2022b. 

#### In final step: 
Run tst_prediction.m, which load, model and predict the targets using MVLR (as the best algorithm resulted in this study). 



