clear variables;
close all;
clc;

%% Read Data

trn_data_input_output  = readtable('./Bitumen/DT prediction of ahwaz_307/Data/trn_input_output.csv');
tst_data_input_output  = readtable('./Bitumen/DT prediction of ahwaz_307/Data/tst_input_output.csv');

path = './Bitumen/DT prediction of ahwaz_307/Data/trn_tst/';

%% Training, and Test data

trn_lbl = table2array(trn_data_input_output(:,3));
trn_dat = table2array(trn_data_input_output(:,1:2));
trn_input_output = table2array(trn_data_input_output);

tst_lbl = table2array(tst_data_input_output(:,3));
tst_dat = table2array(tst_data_input_output(:,1:2));
tst_input_output = table2array(tst_data_input_output);

save(strcat(path,'trn_dat.dat'),'trn_dat','-ascii');
save(strcat(path,'trn_lbl.dat'),'trn_lbl','-ascii');
save(strcat(path,'trn_input_output.dat'),'trn_input_output','-ascii');

save(strcat(path,'tst_dat.dat'),'tst_dat','-ascii');
save(strcat(path,'tst_lbl.dat'),'tst_lbl','-ascii');
save(strcat(path,'tst_input_output.dat'),'tst_input_output','-ascii');