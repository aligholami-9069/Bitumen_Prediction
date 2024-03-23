clear variables;
close all;
clc;

font_size = 12;

path = './Bitumen/Results/RBF/Param_OPT/';
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

i = 0;
params = [];
col_num = [1,2,3,4,5];
csv_file_path = strcat(path,'rbf_param_tuning.csv');
dlmwrite(csv_file_path,col_num,'delimiter',',');

for neurons = 1:2:200 
for spread = 1:1:50

    i = i + 1;

rng('default')

net = newrb(X_trn',y_trn',0,spread,neurons,25);
y_pred = sim(net, X_tst');
y_pred = y_pred';
rbf_mse = immse(y_tst,y_pred);

set(0,'DefaultFigureVisible','off');
[~,~,rbf_r] = postreg(tst_lbl',y_pred');

%********** Result
    params = [
        i,...
        neurons,...
        spread,...
        rbf_mse,...
        rbf_r,...
        ];
    dlmwrite(csv_file_path,params,'delimiter',',','-append');

end
end