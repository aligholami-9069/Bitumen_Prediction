clear variables;
close all;
clc;

font_size = 12;
path = './Bitumen/Results/Neuro Fuzzy/';
data_path = './Bitumen/Data/trn_tst/';

%% Read Net & Test Data

nn_net  = load(strcat(path,'Net/NF_bitumen.mat'));
nf_bitumen = nn_net.NF_bitumen;

tst_dat = table2array(readtable(strcat(data_path,'tst_dat.dat')));
tst_lbl = table2array(readtable(strcat(data_path,'tst_lbl.dat')));
tst_lbl = tst_lbl(~isnan(tst_lbl));

%% Test the Model

y_pred = evalfis(tst_dat, nf_bitumen);

%% Plot & Results

mse = immse(tst_lbl,y_pred);

figure;
[~,~,r] = postreg(tst_lbl',y_pred');
xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
title(sprintf('MSE = %.5f, R = %.5f',mse,r),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'nf_bitumen.png'))

save(strcat(path,'nf_result_bitumen.mat'),'y_pred','r','mse');