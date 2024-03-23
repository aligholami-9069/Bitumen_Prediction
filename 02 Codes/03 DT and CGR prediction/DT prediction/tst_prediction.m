clear variables;
close all;
clc;

font_size = 12;
path_ahwaz = './Bitumen/DT prediction of ahwaz_307/Results/Prediction of Ahwaz_307 (DT)/';
path_sd = './Bitumen/DT prediction of ahwaz_307/Results/Prediction of SD_3 (DT)/';

%% data create

Ahwaz_307_all  = readtable('./Bitumen/Input data/ahvaz_307.csv');
sd_03_all  = readtable('./Bitumen/Input data/sd_03.csv');

path_1 = './Bitumen/DT prediction of ahwaz_307/Data/trn_tst/Prediction/';
path_2 = './Bitumen/DT prediction of ahwaz_307/Data/trn_tst/Prediction/';

ahvaz_307_data = table2array(Ahwaz_307_all);
save(strcat(path_1,'ahvaz_307_data.dat'),'ahvaz_307_data','-ascii');

sd_03_data = table2array(sd_03_all);
save(strcat(path_2,'sd_03_data.dat'),'sd_03_data','-ascii');

%% Read Data & Models (well A)

tst_dat_all_ahwaz = table2array(readtable('./Bitumen/DT prediction of ahwaz_307/Data/trn_tst/Prediction/ahvaz_307_data.dat'));
tst_dat = tst_dat_all_ahwaz(:,[8 11]);
tst_depth = tst_dat_all_ahwaz(:,1);

%% Read Data & Models (well F)

tst_dat_all_sd = table2array(readtable('./Bitumen/DT prediction of ahwaz_307/Data/trn_tst/Prediction/sd_03_data.dat'));
tst_dat_sd = tst_dat_all_sd(:,[6 9]);
tst_depth_sd = tst_dat_all_sd(:,1);
%%
nn_net  = load('./Bitumen/DT prediction of ahwaz_307/Results/Neural Network/Param_OPT/217_net.mat');
nn_dt = nn_net.net_new;

%% Predict & Save

% Neural Network (ahwaz)
nn_y_pred_ahvaz_307 = sim(nn_dt,tst_dat');
nn_y_pred_ahvaz_307 = nn_y_pred_ahvaz_307';

% Neural Network (sd)
nn_y_pred_sd_3 = sim(nn_dt,tst_dat_sd');
nn_y_pred_sd_3 = nn_y_pred_sd_3';

%% Plot (ahwaz)

figure
plot(tst_depth,nn_y_pred_ahvaz_307)
legend('DT','ANN','Location','north')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('Predicted DT (us/f)','fontweight','bold','fontsize',font_size);
title('DT Prediction by ANN (Well #A)','fontweight','bold','fontsize',font_size);
xlim([min(tst_depth) max(tst_depth)])
saveas(gcf,strcat(path_ahwaz,'prediction_dt_nn.png'))

%% Plot (sd)

figure
plot(tst_depth_sd,nn_y_pred_sd_3)
legend('DT','ANN','Location','north')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('Predicted DT (us/f)','fontweight','bold','fontsize',font_size);
title('DT Prediction by ANN (Well #F)','fontweight','bold','fontsize',font_size);
xlim([min(tst_depth_sd) max(tst_depth_sd)])
saveas(gcf,strcat(path_sd,'prediction_dt_nn_sd.png'))