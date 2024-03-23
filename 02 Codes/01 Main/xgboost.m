clear variables;
close all;
clc;

font_size = 12;
path = './Bitumen/Results/XGBoost/';
data_path = './Bitumen/Data/trn_tst/';

%% Read Data

y_pred = table2array(readtable(strcat(path,'xgb_y_pred.dat')));

tst_lbl = table2array(readtable(strcat(data_path,'tst_lbl.dat')));
tst_lbl = tst_lbl(~isnan(tst_lbl));
y_tst = tst_lbl;

%% Plots

mse = immse(y_tst,y_pred);
    
figure;
[~,~,r] = postreg(tst_lbl',y_pred');
xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
title(sprintf('MSE = %.5f, R = %.5f',mse,r),'fontweight','bold','fontsize',font_size);
axis([0 25 0 30]);
saveas(gcf,strcat(path,'xgb_bitumen.png'))

save(strcat(path,'xgb_result_bitumen.mat'),'y_pred','r','mse');