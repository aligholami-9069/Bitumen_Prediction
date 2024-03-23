clear variables;
close all;
clc;

font_size = 12;
path = './Bitumen/Results/all wells toc predictions/';

%% Read Data & Models (ahwaz_307)



data = table2array(readtable('./Bitumen/Data/trn_tst/Prediction/data.dat'));
tst_dat = data(:,2:6);
tst_depth = data(:,1);

nn_net  = load('./Bitumen/Results/Neural Network/Param_OPT/608_net.mat');
nn_bitumen = nn_net.net_new;

fl_net  = load('./Bitumen/Results/Fuzzy Logic/FuzzyResult.mat');
fl_net = fl_net.FuzzyResult;
[~,idx] = min([fl_net.mse]);
fl_net = fl_net(1,idx).Fuzzy;

nf_net  = load('./Bitumen/Results/Neuro Fuzzy/Net/NF_bitumen_1.mat');
nf_bitumen = nf_net.NF_bitumen_1;

rbf_mdl  = load('./Bitumen/Results/RBF/RBFResult.mat');
rbf_mdl = rbf_mdl.RBFResult.model;

xgb_y_pred_test  = load('./Bitumen/Results/XGBoost/xgb_y_pred_test.dat');

lgbm_y_pred_test  = load('./Bitumen/Results/LightGBM/lgbm_y_pred_test.dat');
%% Predict & Save
% Neural Network
nn_y_pred = sim(nn_bitumen,tst_dat');
nn_y_pred = nn_y_pred';

% Fuzzy Logic
fl_y_pred = evalfis(tst_dat,fl_net);

% Neuro Fuzzy
nf_y_pred = evalfis(tst_dat,nf_bitumen);

% RBF
rbf_y_pred = sim(rbf_mdl, tst_dat');
rbf_y_pred = rbf_y_pred';


nn_y_pred = max(nn_y_pred,0);
nn_y_pred = min(nn_y_pred,30);

fl_y_pred = max(fl_y_pred,0);
fl_y_pred = min(fl_y_pred,30);

nf_y_pred = max(nf_y_pred,0);
nf_y_pred = min(nf_y_pred,30);

rbf_y_pred = max(rbf_y_pred,0);
rbf_y_pred = min(rbf_y_pred,30);

xgb_y_pred_test = max(xgb_y_pred_test,0);
xgb_y_pred_test = min(xgb_y_pred_test,30);

lgbm_y_pred_test = max(lgbm_y_pred_test,0);
lgbm_y_pred_test = min(lgbm_y_pred_test,30);

%% Optimization SA

% Optimization weights
w_1_sa = 0.333293236438408;
w_2_sa = 0.00814319346376764;
w_3_sa = 0.103585000503232;
w_4_sa = 0.0000000000000000000000000127933977727791;
w_5_sa = 0.0154004304166284;
w_6_sa = 0.566265730219471;


opt_y_pred_sa = (w_1_sa * nn_y_pred) + (w_2_sa * fl_y_pred) + (w_3_sa * nf_y_pred) + ...
             (w_4_sa * rbf_y_pred) + (w_5_sa * xgb_y_pred_test) + (w_6_sa * lgbm_y_pred_test);
         
opt_y_pred_sa = max(opt_y_pred_sa,0);
opt_y_pred_sa = min(opt_y_pred_sa,30);

y_Predict_Test_sa.ANN = nn_y_pred;
y_Predict_Test_sa.FL = fl_y_pred;
y_Predict_Test_sa.NF = nf_y_pred;
y_Predict_Test_sa.RBF = rbf_y_pred;
y_Predict_Test_sa.XGB = xgb_y_pred_test;
y_Predict_Test_sa.LGBM = lgbm_y_pred_test;
y_Predict_Test_sa.OPT = opt_y_pred_sa;

save(strcat(path,'y_Predict_Test_sa.mat'),'y_Predict_Test_sa');

%% Optimization GA

% Optimization weights
w_1_ga = 0.333273904220964;
w_2_ga = 0.0559616152415831;
w_3_ga = 0.0555092543838138;
w_4_ga = 0.00000667179243241112;
w_5_ga = 0.0154642471554635;
w_6_ga = 0.566468507139957;


opt_y_pred_ga = (w_1_ga * nn_y_pred) + (w_2_ga * fl_y_pred) + (w_3_ga * nf_y_pred) + ...
             (w_4_ga * rbf_y_pred) + (w_5_ga * xgb_y_pred_test) + (w_6_ga * lgbm_y_pred_test);
         
opt_y_pred_ga = max(opt_y_pred_ga,0);
opt_y_pred_ga = min(opt_y_pred_ga,30);

y_Predict_Test_ga.ANN = nn_y_pred;
y_Predict_Test_ga.FL = fl_y_pred;
y_Predict_Test_ga.NF = nf_y_pred;
y_Predict_Test_ga.RBF = rbf_y_pred;
y_Predict_Test_ga.XGB = xgb_y_pred_test;
y_Predict_Test_ga.LGBM = lgbm_y_pred_test;
y_Predict_Test_ga.OPT = opt_y_pred_sa;

save(strcat(path,'y_Predict_Test_ga.mat'),'y_Predict_Test_ga');

%% Optimization ACOR

% Optimization weights
w_1_aco = 0.263439065108514;
w_2_aco = 0.1;
w_3_aco = 0.1;
w_4_aco = 0.1;
w_5_aco = 0.1;
w_6_aco = 0.383789649415693;


opt_y_pred_aco = (w_1_aco * nn_y_pred) + (w_2_aco * fl_y_pred) + (w_3_aco * nf_y_pred) + ...
             (w_4_aco * rbf_y_pred) + (w_5_aco * xgb_y_pred_test) + (w_6_aco * lgbm_y_pred_test);
         
opt_y_pred_aco = max(opt_y_pred_aco,0);
opt_y_pred_aco = min(opt_y_pred_aco,30);

y_Predict_Test_aco.ANN = nn_y_pred;
y_Predict_Test_aco.FL = fl_y_pred;
y_Predict_Test_aco.NF = nf_y_pred;
y_Predict_Test_aco.RBF = rbf_y_pred;
y_Predict_Test_aco.XGB = xgb_y_pred_test;
y_Predict_Test_aco.LGBM = lgbm_y_pred_test;
y_Predict_Test_aco.OPT = opt_y_pred_sa;

save(strcat(path,'y_Predict_Test_aco.mat'),'y_Predict_Test_aco');
%% MVLR

Y_final=(-7.28253611118668) + (1212.87538215154 * opt_y_pred_ga) + (4.90720828837668 * opt_y_pred_aco) + (-1216.34530521682 * opt_y_pred_sa);
Y_final = max(Y_final,0);
Y_final = min(Y_final,30);
save(strcat(path,'Y_final.mat'),'Y_final');

%% Plot (ahwaz)

figure
plot(tst_depth,Y_final)
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('Bitumen (mgHC/g rock)','fontweight','bold','fontsize',font_size);
title('Bitumen Prediction by MVLR (Well: C)','fontweight','bold','fontsize',font_size);
xlim([min(tst_depth) max(tst_depth)])
hold on
scatter(tst_depth,data(:,7),"filled");
legend('Predicted Bitumen','Measured Bitumen','Location','northwest')
axis([4391.1115 4681.7383 0 35]); %well c

saveas(gcf,strcat(path,'prediction_bitumen_cmga_well_c.png'))