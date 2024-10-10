clear variables;
close all;
clc;

font_size = 12;
path = './Bitumen/Results/Three Regression/';

dec_p = 10;

%% Read Data

ga_result  = load('./Bitumen/Results/Optimization/result_y_predict.mat');
ga_y_pred = ga_result.y_Predict_Result.y_pred.GA;

aco_result  = load('./Bitumen/Results/Optimization/result_y_predict.mat');
aco_y_pred = aco_result.y_Predict_Result.y_pred.ACO;

sa_result  = load('./Bitumen/Results/Optimization/result_y_predict.mat');
sa_y_pred = sa_result.y_Predict_Result.y_pred.SA;

cma_result  = load('./Bitumen/Results/Optimization/CMA-ES/CMAESResult.mat');
cma_y_pred = cma_result.CMAESResult.y_pred;

gwo_result  = load('./Bitumen/Results/Optimization/result_y_predict.mat');
gwo_y_pred = gwo_result.y_Predict_Result.y_pred.GWO;

tst_lbl = table2array(readtable('./Bitumen/data/trn_tst/tst_lbl.dat'));
tst_lbl = tst_lbl(~isnan(tst_lbl));
%% Coefficients

x1 = ga_y_pred;
x2 = sa_y_pred;
x3 = aco_y_pred;
x4 = cma_y_pred;
x5 = gwo_y_pred;
y = tst_lbl; 
x=[ones(size(x1)), x1, x2, x3, x4, x5];
a=x\y;
%% Multi-variable linear regression (MVLR)
Y_final=(a(1,1)) + (a(2,1) * x1) + (a(3,1) * x2) + (a(4,1) * x3) + (a(5,1) * x4) + (a(6,1) * x5);
Y_final = max(Y_final,0);
mlr_mse = immse(y,Y_final);

figure;
[~,~,mlr_r] = postreg(y',Y_final');
xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
title 'Optimised by MVLR'
subtitle(sprintf('MSE = %.5f, R = %.5f',mlr_mse,mlr_r),'fontweight','bold','fontsize',font_size);

saveas(gcf,strcat(path,'3_reg.png'))

save(strcat(path,'mlr_result_bitumen.mat'),'Y_final','mlr_r','mlr_mse');