clear variables;
close all;
clc;

font_size = 12;
path_aj140 = './Bitumen/CGR prediction/Results/Prediction of aj 140 (CGR)/';
path_mi06 = './Bitumen/CGR prediction/Results/Prediction of mi06 (CGR)/';

%% data create

aj_140_all  = readtable('./Bitumen/Input data/aj_140.csv');
mi_06_all  = readtable('./Bitumen/Input data/mi_06.csv');

path_1 = './Bitumen/CGR prediction/Data/trn_tst/Prediction/';
path_2 = './Bitumen/CGR prediction/Data/trn_tst/Prediction/';

aj_140_data = table2array(aj_140_all);
save(strcat(path_1,'aj_140_data.dat'),'aj_140_data','-ascii');

mi_06_data = table2array(mi_06_all);
save(strcat(path_2,'mi_06_data.dat'),'mi_06_data','-ascii');

%% Read Data & Models (aj 140)

tst_dat_all_aj140 = table2array(readtable('./Bitumen/CGR prediction/Data/trn_tst/Prediction/aj_140_data.dat'));
tst_dat = tst_dat_all_aj140(:,5);
tst_depth = tst_dat_all_aj140(:,1);

%% Read Data & Models (mi 06)

tst_dat_all_mi06 = table2array(readtable('./Bitumen/CGR prediction/Data/trn_tst/Prediction/mi_06_data.dat'));
tst_dat_mi06 = tst_dat_all_mi06(:,4);
tst_depth_mi06 = tst_dat_all_mi06(:,1);

nn_net  = load('./Bitumen/CGR prediction/Results/Neural Network/Param_OPT/353_net.mat');
nn_dt = nn_net.net_new;

%% Predict & Save

% Neural Network (Well B)
nn_y_pred_aj140 = sim(nn_dt,tst_dat');
nn_y_pred_aj140 = nn_y_pred_aj140';
% Neural Network (Well D)
nn_y_pred_mi_06 = sim(nn_dt,tst_dat_mi06');
nn_y_pred_mi_06 = nn_y_pred_mi_06';


%% Plot (aj 140)

figure
plot(tst_depth,nn_y_pred_aj140)
legend('CGR','ANN','Location','north')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('Predicted CGR (GAPI)','fontweight','bold','fontsize',font_size);
title('CGR Prediction by ANN (Well #B)','fontweight','bold','fontsize',font_size);
xlim([min(tst_depth) max(tst_depth)])
saveas(gcf,strcat(path_aj140,'prediction_cgr_nn_aj140.png'))

%% Plot (mi 06)

figure
plot(tst_depth_mi06,nn_y_pred_mi_06)
legend('CGR','ANN','Location','north')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('Predicted CGR (GAPI)','fontweight','bold','fontsize',font_size);
title('CGR Prediction by ANN (Well #D)','fontweight','bold','fontsize',font_size);
xlim([min(tst_depth_mi06) max(tst_depth_mi06)])
saveas(gcf,strcat(path_mi06,'prediction_cgr_nn_mi06.png'))