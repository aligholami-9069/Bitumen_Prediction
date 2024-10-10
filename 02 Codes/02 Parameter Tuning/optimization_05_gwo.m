clear variables;
close all;
clc;

font_size = 12;
path = './Bitumen/Results/Optimization/GWO/';

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

%% GWO

dim = 6;
lb = 0;
ub = 1;

i = 0;
params = [];
col_num = [1,2,3,4,5];
csv_file_path = strcat(path,'gwo_param_tuning.csv');
dlmwrite(csv_file_path,col_num,'delimiter',',');

for SearchAgents_no = 1:1:1200           % Number of search agents
for Max_iteration = 100:200:900          % Maximum number of iterations
    
    i = i + 1;
    
    rng('default')

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

    set(0,'DefaultFigureVisible','off');
    [~,~,gwo_r] = postreg(tst_lbl',gwo_y');

    %********** Result
    params = [i,SearchAgents_no,Max_iteration,gwo_mse,gwo_r];
    dlmwrite(csv_file_path,params,'delimiter',',','-append','precision',11);

end
end

%% Functions

function f = MSE_WA(w,data1,data2,data3,data4,data5,data6,data_real,N)
    squaredError = (((w(1) * data1) + (w(2) * data2) + (w(3) * data3)+ (w(4) * data4)+ (w(5) * data5)+ (w(6) * data6)) - data_real) .^2;
    f = sum(squaredError(:)) / N;
end

function [Alpha_score,Alpha_pos,Convergence_curve] = GWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,d1,d2,d3,d4,d5,d6,d_real,d_count)

    Alpha_pos=zeros(1,dim);
    Alpha_score=inf; 

    Beta_pos=zeros(1,dim);
    Beta_score=inf; 

    Delta_pos=zeros(1,dim);
    Delta_score=inf; 

    Positions=initialization(SearchAgents_no,dim,ub,lb);

    Convergence_curve=zeros(1,Max_iter);

    l=0;

    while l<Max_iter
        for i=1:size(Positions,1)  

            Flag4ub=Positions(i,:)>ub;
            Flag4lb=Positions(i,:)<lb;
            Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               

            fitness=fobj(Positions(i,:),d1,d2,d3,d4,d5,d6,d_real,d_count);

            if fitness<Alpha_score 
                Alpha_score=fitness; 
                Alpha_pos=Positions(i,:);
            end

            if fitness>Alpha_score && fitness<Beta_score 
                Beta_score=fitness; 
                Beta_pos=Positions(i,:);
            end

            if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score 
                Delta_score=fitness; 
                Delta_pos=Positions(i,:);
            end
        end


        a=2-l*((2)/Max_iter); 

        for i=1:size(Positions,1)
            for j=1:size(Positions,2)
                
                rng('default')

                r1=rand(); 
                r2=rand(); 

                A1=2*a*r1-a; 
                C1=2*r2; 

                D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); 
                X1=Alpha_pos(j)-A1*D_alpha;

                r1=rand();
                r2=rand();

                A2=2*a*r1-a; 
                C2=2*r2; 

                D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); 
                X2=Beta_pos(j)-A2*D_beta;     

                r1=rand();
                r2=rand(); 

                A3=2*a*r1-a; 
                C3=2*r2; 

                D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); 
                X3=Delta_pos(j)-A3*D_delta;       

                Positions(i,j)=(X1+X2+X3)/3;

            end
        end
        l=l+1;    
        Convergence_curve(l)=Alpha_score;
    end
end

function Positions=initialization(SearchAgents_no,dim,ub,lb)

rng('default')

Boundary_no= size(ub,2); 
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end