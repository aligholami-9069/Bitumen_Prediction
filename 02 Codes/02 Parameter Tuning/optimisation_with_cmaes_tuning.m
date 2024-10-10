clear variables;
close all;
clc;

path = './Bitumen/Results/Optimization/CMA-ES/';
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

%% CMAES


rng('default')
nVar = 6;                


i = 0;
params = [];
col_num = [1,2,3,4,5,6];
csv_file_path = strcat(path,'cmaes_param_tuning.csv');
dlmwrite(csv_file_path,col_num,'delimiter',',');

for VarMin = 0.1:0.1:0.5           %0  % Lower Bound of Decision Variables
for VarMax = 0.5:0.1:1           %1   % Upper Bound of Decision Variables
for MaxIt = 1:1:300           %300   % Maximum Number of Iterations


    i = i + 1;

rng('default')


lambda = (4+round(3*log(nVar)))*10;  % Population Size (and Number of Offsprings)
mu = round(lambda/2);     % Number of Parents

% Problem
CostFunction=@MSE_WA;   % Cost Function

% Run CMA-ES
[BestSol, BestCost] = CMAES(CostFunction, nVar, VarMin, VarMax, MaxIt, lambda, mu,...
                    nn_y_pred,fl_y_pred,nf_y_pred,rbf_y_pred,xgb_y_pred,lgbm_y_pred,tst_lbl,size(tst_lbl,1));

cmaes_x = BestSol.Position;

cmaes_y = (cmaes_x(1) * nn_y_pred)+...
                    (cmaes_x(2) * fl_y_pred)+...
                    (cmaes_x(3) * nf_y_pred)+...
                    (cmaes_x(4) * rbf_y_pred)+...
                    (cmaes_x(5) * xgb_y_pred)+...
                    (cmaes_x(6) * lgbm_y_pred);
                    
                
                
 cmaes_y = max(cmaes_y,0);

cmaes_mse = immse(tst_lbl,cmaes_y);
[~,~,cmaes_r] = postreg(tst_lbl',cmaes_y');


%********** Result
params = [i,VarMin,VarMax,MaxIt,cmaes_mse,cmaes_r];
dlmwrite(csv_file_path,params,'delimiter',',','-append','precision',11);


end
end
end





function f = MSE_WA(w,data1,data2,data3,data4,data5,data6,data_real,N)
    squaredError = (((w(1) * data1) + (w(2) * data2) + (w(3) * data3) + (w(4) * data4) + (w(5) * data5) + (w(6) * data6)) - data_real) .^2;
    f = sum(squaredError(:)) / N;
end


% CMA-ES Function
function [BestSol, BestCost] = CMAES(CostFunction, nVar, VarMin, VarMax, MaxIt, lambda, mu, d1, d2, d3, d4, d5, d6, d_real, d_count)
    % Parameters
    VarSize = [1 nVar];       % Decision Variables Matrix Size
    sigma0 = 0.3*(VarMax-VarMin);  % Initial Step Size
    w = log(mu+0.5)-log(1:mu); % Parent Weights
    w = w/sum(w); 
    mu_eff = 1/sum(w.^2);      % Number of Effective Solutions
    cs = (mu_eff+2)/(nVar+mu_eff+5); % Step Size Control Parameters
    ds = 1+cs+2*max(sqrt((mu_eff-1)/(nVar+1))-1,0);
    ENN = sqrt(nVar)*(1-1/(4*nVar)+1/(21*nVar^2));
    cc = (4+mu_eff/nVar)/(4+nVar+2*mu_eff/nVar);  % Covariance Update Parameters
    c1 = 2/((nVar+1.3)^2+mu_eff);
    alpha_mu = 2;
    cmu = min(1-c1,alpha_mu*(mu_eff-2+1/mu_eff)/((nVar+2)^2+alpha_mu*mu_eff/2));
    hth = (1.4+2/(nVar+1))*ENN;

    % Initialization
    ps = cell(MaxIt,1);
    pc = cell(MaxIt,1);
    C = cell(MaxIt,1);
    sigma = cell(MaxIt,1);
    ps{1} = zeros(VarSize);
    pc{1} = zeros(VarSize);
    C{1} = eye(nVar);
    sigma{1} = sigma0;

    empty_individual.Position = [];
    empty_individual.Step = [];
    empty_individual.Cost = [];

    M = repmat(empty_individual, MaxIt, 1);
    M(1).Position = unifrnd(VarMin, VarMax, VarSize);
    M(1).Step = zeros(VarSize);
    M(1).Cost = CostFunction(M(1).Position, d1, d2, d3, d4, d5, d6, d_real, d_count);

    BestSol = M(1);
    BestCost = zeros(MaxIt,1);

    % CMA-ES Main Loop
    for g = 1:MaxIt
        % Generate Samples
        pop = repmat(empty_individual, lambda, 1);
        for i = 1:lambda
            pop(i).Step = mvnrnd(zeros(VarSize), C{g});
            pop(i).Position = M(g).Position + sigma{g}*pop(i).Step;
            pop(i).Cost = CostFunction(pop(i).Position, d1, d2, d3, d4, d5, d6, d_real, d_count);

            % Update Best Solution Ever Found
            if pop(i).Cost < BestSol.Cost
                BestSol = pop(i);
            end
        end

        % Sort Population
        Costs = [pop.Cost];
        [Costs, SortOrder] = sort(Costs);
        pop = pop(SortOrder);

        % Save Results
        BestCost(g) = BestSol.Cost;

        % Display Results
        disp(['Iteration ' num2str(g) ': Best Cost = ' num2str(BestCost(g))]);

        % Exit At Last Iteration
        if g == MaxIt
            break;
        end

        % Update Mean
        M(g+1).Step = 0;
        for j = 1:mu
            M(g+1).Step = M(g+1).Step + w(j)*pop(j).Step;
        end
        M(g+1).Position = M(g).Position + sigma{g}*M(g+1).Step;
        M(g+1).Cost = CostFunction(M(g+1).Position, d1, d2, d3, d4, d5, d6, d_real, d_count);
        if M(g+1).Cost < BestSol.Cost
            BestSol = M(g+1);
        end

        % Update Step Size
        ps{g+1} = (1-cs)*ps{g} + sqrt(cs*(2-cs)*mu_eff)*M(g+1).Step/chol(C{g})';
        sigma{g+1} = sigma{g}*exp(cs/ds*(norm(ps{g+1})/ENN-1))^0.3;

        % Update Covariance Matrix
        if norm(ps{g+1})/sqrt(1-(1-cs)^(2*(g+1))) < hth
            hs = 1;
        else
            hs = 0;
        end
        delta = (1-hs)*cc*(2-cc);
        pc{g+1} = (1-cc)*pc{g} + hs*sqrt(cc*(2-cc)*mu_eff)*M(g+1).Step;
        C{g+1} = (1-c1-cmu)*C{g} + c1*(pc{g+1}'*pc{g+1} + delta*C{g});
        for j = 1:mu
            C{g+1} = C{g+1} + cmu*w(j)*pop(j).Step'*pop(j).Step;
        end

        % If Covariance Matrix is not Positive Definite or Near Singular
        [V, E] = eig(C{g+1});
        if any(diag(E) < 0)
            E = max(E, 0);
            C{g+1} = V*E/V;
        end
    end
end


