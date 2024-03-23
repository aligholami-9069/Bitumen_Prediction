clear variables;
close all;
clc;

font_size = 12;
path = './Bitumen/DT prediction of ahwaz_307/Results/Neural Network/';

%% Read Net & Test Data

nn_net  = load(strcat(path,'Param_OPT/217_net.mat'));

nn_dt = nn_net.net_new;

tst_dat = table2array(readtable('./Bitumen/DT prediction of ahwaz_307/Data/trn_tst/tst_dat.dat'));
tst_lbl = table2array(readtable('./Bitumen/DT prediction of ahwaz_307/Data/trn_tst/tst_lbl.dat'));
tst_lbl = tst_lbl(~isnan(tst_lbl));

%% Test the Model

y_pred = sim(nn_dt,tst_dat');
y_pred = y_pred';

%% Plot & Results

mse = immse(tst_lbl,y_pred);

figure;
[~,~,r] = postreg(tst_lbl',y_pred');
xlabel(strcat('DT (Predicted)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DT (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('MSE = %.5f, R = %.5f',mse,r),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'nn_dt.png'))

save(strcat(path,'nn_result_dt.mat'),'y_pred','r','mse');