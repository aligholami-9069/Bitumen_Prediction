clear variables;
close all;
clc;

font_size = 12;


%% ANN
path = './Bitumen/DT prediction of ahwaz_307/Results/Neural Network/Param_OPT/';
data = readtable(strcat(path,'ANN_all_plot.xlsx'));

x1 = table2array(data(:,1));
y1 = table2array(data(:,2));

z1 = table2array(data(:,3));
z2 = table2array(data(:,4));

figure;
scatter3(x1,z1,y1,'.')
xlabel('# of models','fontweight','bold','fontsize',font_size)
ylabel('MSE','fontweight','bold','fontsize',font_size)
zlabel('No. of Neurons','fontweight','bold','fontsize',font_size)
title('ANN Parameter Tuning for DT Prediction','fontweight','bold','fontsize',font_size)
saveas(gcf,strcat(path,'ann_mse_3d_plot.png'))

figure;
scatter3(x1,z2,y1,'.','red')
xlabel('# of models','fontweight','bold','fontsize',font_size)
ylabel('R','fontweight','bold','fontsize',font_size)
zlabel('No. of Neurons','fontweight','bold','fontsize',font_size)
title('ANN Parameter Tuning for DT Prediction','fontweight','bold','fontsize',font_size)
saveas(gcf,strcat(path,'ann_r_3d_plot.png'))
