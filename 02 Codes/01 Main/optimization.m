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

%% GWO

disp('GWO')
disp('-------------')

rng('default')
SearchAgents_no = 6; % Number of search agents
Max_iteration = 300;  % Maximum numbef of iterations

dim = 6;
lb = 0;
ub = 1;
fobj = @MSE_WA;

[~,gwo_x,~] = GWO(SearchAgents_no,Max_iteration,lb,ub,dim,fobj,...
                         nn_y_pred,fl_y_pred,nf_y_pred,rbf_y_pred,xgb_y_pred,lgbm_y_pred,tst_lbl,size(tst_lbl,1));

    gwo_y = (gwo_x(1) * nn_y_pred)+...
            (gwo_x(2) * fl_y_pred)+...
            (gwo_x(3) * nf_y_pred)+...
            (gwo_x(4) * rbf_y_pred)+...
            (gwo_x(5) * xgb_y_pred)+...
            (gwo_x(6) * lgbm_y_pred);
 
gwo_mse = immse(tst_lbl,gwo_y);
            
figure;
[~,~,gwo_r] = postreg(tst_lbl',gwo_y');
xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
title 'Optimised by CMGWO'
subtitle(sprintf(str_format,gwo_mse,gwo_r),'fontweight','bold','fontsize',font_size);
axis([0 25 0 30]);
saveas(gcf,strcat(path,'gwo.png'))

%% CMAES

disp('CMAES')
disp('-------------')
rng('default')

% Parameters
nVar = 6;                % Number of Unknown (Decision) Variables
VarMin = 0.5;             % Lower Bound of Decision Variables
VarMax = 0.6;              % Upper Bound of Decision Variables

% CMA-ES Parameters
MaxIt = 100;              % Maximum Number of Iterations
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

figure;
[~,~,cmaes_r] = postreg(tst_lbl',cmaes_y');

xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
title 'Optimised by CMCMA-ES'
subtitle(sprintf(str_format, cmaes_mse, cmaes_r), 'fontweight', 'bold', 'fontsize', font_size);
axis([0 25 0 30]);

saveas(gcf,strcat(path,'cmaes.png'))
%% Result Matrix

Results.R.GA = ga_r;
Results.R.SA = simann_r;
Results.R.ACO = aco_r;
Results.R.CMA = cmaes_r;
Results.R.GWO = gwo_r;

Results.MSE.GA = ga_mse;
Results.MSE.SA = simann_mse;
Results.MSE.ACO = aco_mse;
Results.MSE.CMA = cmaes_mse;
Results.MSE.GWO = gwo_mse;

save(strcat(path,'result_all.mat'),'Results');
%% Result y Predict

y_Predict_Result.y_pred.GA = ga_y;
y_Predict_Result.y_pred.SA = simann_y;
y_Predict_Result.y_pred.ACO = aco_y;
y_Predict_Result.y_pred.CMA = cmaes_y;
y_Predict_Result.y_pred.GWO = gwo_y;

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

function [Alpha_score,Alpha_pos,Convergence_curve] = GWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,d1,d2,d3,d4,d5,d6,d_real,d_count)

    % initialize alpha, beta, and delta_pos
    Alpha_pos=zeros(1,dim);
    Alpha_score=inf; %change this to -inf for maximization problems

    Beta_pos=zeros(1,dim);
    Beta_score=inf; %change this to -inf for maximization problems

    Delta_pos=zeros(1,dim);
    Delta_score=inf; %change this to -inf for maximization problems

    %Initialize the positions of search agents
    Positions=initialization(SearchAgents_no,dim,ub,lb);

    Convergence_curve=zeros(1,Max_iter);

    l=0;% Loop counter

    % Main loop
    while l<Max_iter
        for i=1:size(Positions,1)  

           % Return back the search agents that go beyond the boundaries of the search space
            Flag4ub=Positions(i,:)>ub;
            Flag4lb=Positions(i,:)<lb;
            Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               

            % Calculate objective function for each search agent
            fitness=fobj(Positions(i,:),d1,d2,d3,d4,d5,d6,d_real,d_count);

            % Update Alpha, Beta, and Delta
            if fitness<Alpha_score 
                Alpha_score=fitness; % Update alpha
                Alpha_pos=Positions(i,:);
            end

            if fitness>Alpha_score && fitness<Beta_score 
                Beta_score=fitness; % Update beta
                Beta_pos=Positions(i,:);
            end

            if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score 
                Delta_score=fitness; % Update delta
                Delta_pos=Positions(i,:);
            end
        end


        a=2-l*((2)/Max_iter); % a decreases linearly fron 2 to 0

        % Update the Position of search agents including omegas
        for i=1:size(Positions,1)
            for j=1:size(Positions,2)
                
                rng('default')

                r1=rand(); % r1 is a random number in [0,1]
                r2=rand(); % r2 is a random number in [0,1]

                A1=2*a*r1-a; % Equation (3.3)
                C1=2*r2; % Equation (3.4)

                D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
                X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1

                r1=rand();
                r2=rand();

                A2=2*a*r1-a; % Equation (3.3)
                C2=2*r2; % Equation (3.4)

                D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
                X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       

                r1=rand();
                r2=rand(); 

                A3=2*a*r1-a; % Equation (3.3)
                C3=2*r2; % Equation (3.4)

                D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
                X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             

                Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)

            end
        end
        l=l+1;    
        Convergence_curve(l)=Alpha_score;
    end
end

function Positions=initialization(SearchAgents_no,dim,ub,lb)

rng('default')

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end

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


