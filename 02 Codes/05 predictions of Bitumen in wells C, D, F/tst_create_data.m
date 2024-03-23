clear variables;
close all;
clc;

%%

data_all  = readtable('C:/Users/Ali/Desktop/Bitumen/final input data/well_c.csv'); %specialize your well data (inputs) to predict bitumen values

path_1 = 'C:/Users/Ali/Desktop/Bitumen/Data/trn_tst/Prediction/';

data = table2array(data_all);
tst_dat = data(:,2:6);
tst_depth = data(:,1);


save(strcat(path_1,'data.dat'),'data','-ascii');
save(strcat(path_1,'tst_dat.dat'),'tst_dat','-ascii');
save(strcat(path_1,'tst_depth.dat'),'tst_depth','-ascii');