clear variables;
close all;
clc;

font_size = 12;
path = './Bitumen/Results/Optimization/GA/';

%% Read Data

nn_result  = load('./Bitumen/Results/Neural Network/nn_result_toc.mat');
nn_y_pred = nn_result.y_pred;

fl_y_pred  = load('./Bitumen/Results/Fuzzy Logic/FuzzyResult.mat');
fl_y_pred = fl_y_pred.FuzzyResult;
[~,idx] = min([fl_y_pred.mse]);
fl_y_pred = fl_y_pred(1,idx).y_pred;

nf_result  = load('./Bitumen/Results/Neuro Fuzzy/nf_result_bitumen.mat');
nf_y_pred = nf_result.y_pred;

rbf_result  = load('./Bitumen/Results/RBF/RBFResult.mat');
rbf_y_pred = rbf_result.RBFResult.y_pred;

xgb_result  = load('./Bitumen/Results/XGBoost/xgb_result_bitumen.mat');
xgb_y_pred = xgb_result.y_pred;

lgbm_result  = load('./Bitumen/Results/LightGBM/lgbm_result_bitumen.mat');
lgbm_y_pred = lgbm_result.y_pred;

tst_lbl = table2array(readtable('./Bitumen/data/trn_tst/tst_lbl.dat'));
tst_lbl = tst_lbl(~isnan(tst_lbl));
%% GA Algorithm

i = 0;
params = [];
col_num = [1,2,3];
csv_file_path = strcat(path,'ga_param_tuning.csv');
dlmwrite(csv_file_path,col_num,'delimiter',',');

for MaxGenerations = 50:50:500 
for PopulationSize = 50:50:500 
for EliteCount = ceil(0.05 * PopulationSize)
for CrossoverFraction = 0.4:0.2:1
for MutationFcn = {'mutationgaussian','mutationadaptfeasible','mutationuniform'}
for CrossoverFcn = {'crossoverscattered','crossoversinglepoint'}
for SelectionFcn = {'selectionstochunif','selectionroulette'}
for HybridFcn = {'patternsearch','fmincon'}
for InitialPopulationRange = [0;1]

    
    try
        i = i + 1;

        rng('default')
        nvars = 6;
        lb = [0,0,0,0,0,0];
        ub = [1,1,1,1,1,1];
        
            options = optimoptions('ga',...
                'CrossoverFcn',str2func(CrossoverFcn{1}),...
                'CrossoverFraction',CrossoverFraction,...
                'PopulationSize',PopulationSize,...
                'EliteCount',EliteCount,...
                'HybridFcn',str2func(HybridFcn{1}),...
                'InitialPopulationRange',InitialPopulationRange,...
                'MaxGenerations',MaxGenerations,...
                'MutationFcn',str2func(MutationFcn{1}),...
                'SelectionFcn',str2func(SelectionFcn{1}));          

        
        
        fun = @(x) MSE(x(1),x(2),x(3),x(4),x(5),x(6),...
                         nn_y_pred,fl_y_pred,nf_y_pred,rbf_y_pred,xgb_y_pred,lgbm_y_pred,...
                         tst_lbl,size(tst_lbl,1));

        ga_x = ga(fun,nvars,[],[],[],[],lb,ub,[],options);

        ga_y = (ga_x(1) * nn_y_pred)+...
               (ga_x(2) * fl_y_pred)+...
               (ga_x(3) * nf_y_pred)+...
               (ga_x(4) * rbf_y_pred)+...
               (ga_x(5) * xgb_y_pred)+...
               (ga_x(6) * lgbm_y_pred);

        ga_mse = immse(tst_lbl,ga_y);


        set(0,'DefaultFigureVisible','off');
        [~,~,ga_r] = postreg(tst_lbl',ga_y');
        
    catch ME
        warning(ME.message);
    end

        %********** Result
        params = [i,...
            CrossoverFcn,...
            CrossoverFraction,...
            PopulationSize,...
            EliteCount,...
            HybridFcn,...
            InitialPopulationRange,...
            MaxGenerations,...
            MutationFcn,...
            SelectionFcn,... 
            ga_mse,ga_r];


        save(strcat(path,"Parameters/",int2str(i),".mat"),'params');
        
        params_csv = [i,ga_mse,ga_r];
        dlmwrite(csv_file_path,params_csv,'delimiter',',','-append','precision',11);
end
end
end
end
end
end
end
end
end
        
%% Functions

function f = MSE(w1,w2,w3,w4,w5,w6,data_x1,data_x2,data_x3,data_x4,data_x5,data_x6,data_real,N)
    squaredError = (((w1 * data_x1) + (w2 * data_x2) + (w3 * data_x3) + (w4 * data_x4) + (w5 * data_x5) + (w6 * data_x6)) - data_real) .^2;
    f = sum(squaredError(:)) / N;
end