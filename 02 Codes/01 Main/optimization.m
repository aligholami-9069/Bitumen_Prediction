clear variables;
close all;
clc;

font_size = 12;
path = './Bitumen/Results/Optimization/';

dec_p = 6;
str_format = strcat('MSE = %.',num2str(dec_p),'f, R = %.',num2str(dec_p),'f');

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

%% GA

disp('GA')
disp('-------------')

rng('default')
nvars = 6;
lb = [0,0,0,0,0,0];
ub = [1,1,1,1,1,1];

options = optimoptions('ga',...
            'CrossoverFcn',str2func('crossoverscattered'),...
            'CrossoverFraction',1,...
            'PopulationSize',450,...
            'EliteCount',23,...
            'HybridFcn',str2func('patternsearch'),...
            'InitialPopulationRange',[0;1],...
            'MaxGenerations',500,...
            'MutationFcn',str2func('mutationadaptfeasible'),...
            'MigrationFraction',0.2,...
            'SelectionFcn',str2func('selectionroulette'));


        
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

figure;
[~,~,ga_r] = postreg(tst_lbl',ga_y');
xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
title 'Optimised by CMGA'
subtitle(sprintf(str_format,ga_mse,ga_r),'fontweight','bold','fontsize',font_size);
axis([0 25 0 30]);
saveas(gcf,strcat(path,'ga.png'))

%% SA

disp('SA')
disp('-------------')

rng('default')
x0 = [0,0,0,0,0,0];
lb = [0,0,0,0,0,0];
ub = [1,1,1,1,1,1];
options = optimoptions('simulannealbnd',...
            'InitialTemperature',4000,...
            'ReannealInterval',6500,...           
            'FunctionTolerance',1e-6,...
            'HybridFcn',str2func('fminsearch'),...
            'HybridInterval','never',...
            'TemperatureFcn',str2func('temperatureexp'),...
            'AnnealingFcn',str2func('annealingboltz'));


fun = @(x) MSE(x(1),x(2),x(3),x(4),x(5),x(6),...
            nn_y_pred,fl_y_pred,nf_y_pred,rbf_y_pred,xgb_y_pred,lgbm_y_pred,...
            tst_lbl,size(tst_lbl,1));
                     
simann_x = simulannealbnd(fun,x0,lb,ub,options);

simann_y = (simann_x(1) * nn_y_pred)+...
           (simann_x(2) * fl_y_pred)+...
           (simann_x(3) * nf_y_pred)+...
           (simann_x(4) * rbf_y_pred)+...
           (simann_x(5) * xgb_y_pred)+...
           (simann_x(6) * lgbm_y_pred);

simann_mse = immse(tst_lbl,simann_y);
            
figure;
[~,~,simann_r] = postreg(tst_lbl',simann_y');
xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
title 'Optimised by CMSA'
subtitle(sprintf(str_format,simann_mse,simann_r),'fontweight','bold','fontsize',font_size);
axis([0 25 0 30]);
saveas(gcf,strcat(path,'simann.png'))

%% ACO

disp('ACO')
disp('-------------')

rng('default')
n_iter = 200; %number of iteration
NA = 500; % Number of Ants
alpha = 1; % alpha
beta = 0.5; % beta
roh = 0.4; % Evaporation rate
n_param = 6; % Number of paramters
LB = [0.1,0.1,0.1,0.1,0.1,0.1]; % lower bound
UB = [0.99,0.99,0.99,0.99,0.99,0.99]; % upper bound

n_node = 600; % numbe

aco_x = ACO(nn_y_pred,fl_y_pred,nf_y_pred,rbf_y_pred,xgb_y_pred,lgbm_y_pred,tst_lbl,size(tst_lbl,1),...
                            n_iter,NA,alpha,beta,roh,n_param,LB,UB,n_node);
             
aco_y = (aco_x(1) * nn_y_pred)+...
        (aco_x(2) * fl_y_pred)+...
        (aco_x(3) * nf_y_pred)+...
        (aco_x(4) * rbf_y_pred)+...
        (aco_x(5) * xgb_y_pred)+...
        (aco_x(6) * lgbm_y_pred);

aco_mse = immse(tst_lbl,aco_y);

figure;
[~,~,aco_r] = postreg(tst_lbl',aco_y');
xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
title 'Optimised by CMACOR'
subtitle(sprintf(str_format,aco_mse,aco_r),'fontweight','bold','fontsize',font_size);
axis([0 25 0 30]);
saveas(gcf,strcat(path,'aco.png'))
%% Result Matrix

Results.R.GA = ga_r;
Results.R.SA = simann_r;
Results.R.ACO = aco_r;

Results.MSE.GA = ga_mse;
Results.MSE.SA = simann_mse;
Results.MSE.ACO = aco_mse;

save(strcat(path,'result_all.mat'),'Results');
%% Result y Predict

y_Predict_Result.y_pred.GA = ga_y;
y_Predict_Result.y_pred.SA = simann_y;
y_Predict_Result.y_pred.ACO = aco_y;

save(strcat(path,'result_y_predict.mat'),'y_Predict_Result');

%% Functions

function f = MSE(w1,w2,w3,w4,w5,w6,data_x1,data_x2,data_x3,data_x4,data_x5,data_x6,data_real,N)
    squaredError = (((w1 * data_x1) + (w2 * data_x2) + (w3 * data_x3) + (w4 * data_x4) + (w5 * data_x5) + (w6 * data_x6)) - data_real) .^2;
    f = sum(squaredError(:)) / N;
end

function f = MSE_WA(w,data1,data2,data3,data4,data5,data6,data_real,N)
    squaredError = (((w(1) * data1) + (w(2) * data2) + (w(3) * data3) + (w(4) * data4) + (w(5) * data5) + (w(6) * data6)) - data_real) .^2;
    f = sum(squaredError(:)) / N;
end


function opt_params = ACO(d1,d2,d3,d4,d5,d6,d_real,d_count,n_iter,NA,alpha,beta,roh,n_param,LB,UB,n_node)
   
    % intializing some variables 
    cost_best_prev = inf;
    ant = zeros(NA,n_param);
    cost = zeros(NA,1);
    tour_selected_param = zeros(1,n_param);
    Nodes = zeros(n_node,n_param);
    prob = zeros(n_node, n_param);

    % Generating Nodes
    T = ones(n_node,n_param).*eps; % Phormone Matrix
    dT = zeros(n_node,n_param); % Change of Phormone
    for i = 1:n_param
        Nodes(:,i) = linspace(LB(i),UB(i),n_node); % Node generation at equal spaced points
    end

    % Iteration loop
    for iter = 1:n_iter
        for tour_i = 1:n_param
            prob(:,tour_i) = (T(:,tour_i).^alpha) .* ((1./Nodes(:,tour_i)).^beta);
            prob(:,tour_i) = prob(:,tour_i)./sum(prob(:,tour_i));
        end
        for A = 1:NA
            for tour_i = 1:n_param
                node_sel = rand;
                node_ind = 1;
                prob_sum = 0;
                for j = 1:n_node
                    prob_sum = prob_sum + prob(j,tour_i);
                    if prob_sum >= node_sel
                        node_ind = j;
                        break
                    end
                end
                ant(A,tour_i) = node_ind;
                tour_selected_param(tour_i) = Nodes(node_ind, tour_i);
            end
            cost(A) = MSE_WA(tour_selected_param,d1,d2,d3,d4,d5,d6,d_real,d_count);
            
            if iter ~= 1
%                 
                for i = 1:n_param
                    tour_selected_param(i) = Nodes(ant(cost_best_ind,i),i);
                end
            end
        end
        [cost_best,cost_best_ind] = min(cost);

        % Elitsem
        if (cost_best>cost_best_prev) && (iter ~= 1)
            [~,cost_worst_ind] = max(cost);
            ant(cost_worst_ind,:) = best_prev_ant;
            cost_best = cost_best_prev;
            cost_best_ind = cost_worst_ind;
        else
            cost_best_prev = cost_best;
            best_prev_ant = ant(cost_best_ind,:);
        end
        dT = zeros(n_node,n_param); % Change of Phormone
        for tour_i = 1:n_param
            for A = 1:NA
                dT(ant(A,tour_i),tour_i) = dT(ant(A,tour_i),tour_i) + cost_best/cost(A);
            end
        end
        T = roh.*T + dT;        
    end   
    opt_params = tour_selected_param;
end
