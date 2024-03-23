clear variables;
close all;
clc;

font_size = 12;
path = './Bitumen/Results/Neural Network/';

%% Read Net & Test Data


nn_net  = load(strcat(path,'Param_OPT/608_net.mat'));

nn_fva = nn_net.net_new;

tst_dat = table2array(readtable('./Bitumen/Data/trn_tst/tst_dat.dat'));
tst_lbl = table2array(readtable('./Bitumen/Data/trn_tst/tst_lbl.dat'));
tst_lbl = tst_lbl(~isnan(tst_lbl));

%% Test the Model

y_pred = sim(nn_fva,tst_dat');
y_pred = y_pred';
y_pred = max(y_pred,0);

%% Plot & Results

mse = immse(tst_lbl,y_pred);

figure;
[~,~,r] = postreg(tst_lbl',y_pred');
xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
title(sprintf('MSE = %.5f, R = %.5f',mse,r),'fontweight','bold','fontsize',font_size);
axis([0 25 0 30]);
saveas(gcf,strcat(path,'nn_toc.png'))

save(strcat(path,'nn_result_toc.mat'),'y_pred','r','mse');


