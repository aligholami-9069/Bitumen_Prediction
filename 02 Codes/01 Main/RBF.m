clear variables;
close all;
clc;

font_size = 12;

path = './Bitumen/Results/RBF/';
data_path = './Bitumen/Data/trn_tst/';
%% Read Data

trn_dat = table2array(readtable(strcat(data_path,'trn_dat.dat')));
trn_lbl = table2array(readtable(strcat(data_path,'trn_lbl.dat')));
trn_lbl = trn_lbl(~isnan(trn_lbl));

tst_dat = table2array(readtable(strcat(data_path,'tst_dat.dat')));
tst_lbl = table2array(readtable(strcat(data_path,'tst_lbl.dat')));
tst_lbl = tst_lbl(~isnan(tst_lbl));

X_trn = trn_dat;
X_tst = tst_dat;
y_trn = trn_lbl;
y_tst = tst_lbl;
%% RBF

rng('default')

neurons = 27; 
spread = 6;
Neurons_between_displays = 25;

net = newrb(X_trn',y_trn',0,spread,neurons,Neurons_between_displays);
y_pred = sim(net, X_tst');
y_pred = y_pred';
%%
mse = immse(y_tst,y_pred);

figure;
[~,~,r] = postreg(tst_lbl',y_pred');
xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
title(sprintf('MSE = %.5f, R = %.5f',mse,r),'fontweight','bold','fontsize',font_size);
axis([0 25 0 30]);
saveas(gcf,strcat(path,'rbf_bitumen.png'))

RBFResult.model = net;
RBFResult.r = r;
RBFResult.y_pred = y_pred;
RBFResult.mse = mse;

save(strcat(path,'RBFResult.mat'),'RBFResult');
