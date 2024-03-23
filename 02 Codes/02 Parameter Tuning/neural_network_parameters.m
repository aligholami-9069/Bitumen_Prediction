clear variables;
close all;
clc;

font_size = 12;
path = './Bitumen/Results/Neural Network/Param_OPT/';

%% Read Data

trn_dat = table2array(readtable('./Bitumen/Data/trn_tst/trn_dat.dat'));

trn_lbl = table2array(readtable('./Bitumen/Data/trn_tst/trn_lbl.dat'));
trn_lbl = trn_lbl(~isnan(trn_lbl));

tst_dat = table2array(readtable('./Bitumen/Data/trn_tst/tst_dat.dat'));
tst_lbl = table2array(readtable('./Bitumen/Data/trn_tst/tst_lbl.dat'));
tst_lbl = tst_lbl(~isnan(tst_lbl));

x = trn_dat';
t = trn_lbl';

%% NN

%%%%%%%%%%%%%%%%%%%%
rng('default')
trainFcn = 'trainlm';	% 'trainlm','trainbr','trainscg'
%%%%%%%%%%%%%%%%%%%%

i = 0;
params = [];
col_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14];
csv_file_path = strcat(path,'nn_param_tuning.csv');
dlmwrite(csv_file_path,col_num,'delimiter',',');


for hiddenLayerSize = 8:1:30
    
    hiddenLayerSize_2 = hiddenLayerSize;
    hiddenLayerSize_3 = hiddenLayerSize;

    
%     net = fitnet(hiddenLayerSize,trainFcn);
    net = fitnet([hiddenLayerSize,hiddenLayerSize_2,hiddenLayerSize_3],trainFcn);
    
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};
    net.divideFcn = 'dividerand';
    net.divideMode = 'sample';
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', 'plotregression', 'plotfit'};
    net.trainParam.show = 25;                 % Epochs between displays (NaN for no displays). The default value is 25.
    net.trainParam.showCommandLine = false;   % Generate command-line output. The default value is false.
    net.trainParam.showWindow = true;         % Show training GUI. The default value is true.
    net.trainParam.time = inf;                % Maximum time to train in seconds. The default value is inf.
    
for epochs = 1000
    net.trainParam.epochs = epochs;	% Maximum number of epochs to train. The default value is 1000.
    
for goal = 0    
    net.trainParam.goal = goal;	% Performance goal. The default value is 0.
    
for min_grad = 1e-6    
    net.trainParam.min_grad = min_grad;	% Minimum performance gradient. The default value is 1e-6.
    
for max_fail = 4:1:10 %6    
    net.trainParam.max_fail = max_fail; % Maximum validation failures. The default value is 6.

for mu = 0.005
    net.trainParam.mu = mu;	% Marquardt adjustment parameter. The default value is 0.005.
    
for sigma = 5.0e-5
    net.trainParam.sigma = sigma;	% Determine change in weight for second derivative approximation. The default value is 5.0e-5.
   
for lambda = 5.0e-7
    net.trainParam.lambda = lambda;	% Parameter for regulating the indefiniteness of the Hessian. The default value is 5.0e-7.
    
for trainRatio = 0.5:0.1:0.8 %70/100
    net.divideParam.trainRatio = trainRatio;
    valRatio = (1-trainRatio)/2;
    testRatio = (1-trainRatio)/2;
    net.divideParam.valRatio = valRatio;
    net.divideParam.testRatio = testRatio;
       
    i = i + 1;
    
    net.performFcn = 'mse';
                %	'mae'           % Mean absolute error performance function.
                %	'mse'           % Mean squared error performance function.
                %	'sae'           % Sum squared error performance function.
                %	'sse'           % Sum squared error performance function.
                %	'crossentropy'  % Cross-entropy performance.
                %	'msesparse'     % Mean squared error performance function with L2 weight and sparsity regularizers.
        
    % Train the Network
    [net_new,tr] = train(net,x,t);
    
    % Test the Network
    y = net_new(x);
    e = gsubtract(t,y);
    performance = perform(net_new,t,y);

    % Recalculate Training, Validation and Test Performance
    trainTargets = t .* tr.trainMask{1};
    valTargets = t .* tr.valMask{1};
    testTargets = t .* tr.testMask{1};

    % View the Network
    view(net)

    % Plots
    figure;
    plotperform(tr)
    saveas(gcf,strcat(path,'Figures/',num2str(i),'_perform.png'))
    plotregression(trainTargets,y,'Training',valTargets,y,'Validation',testTargets,y,'Test',t,y,'All')
    saveas(gcf,strcat(path,'Figures/',num2str(i),'_regression.png'))
    
    save(strcat(path,'Nets/',num2str(i),'_net.mat'),'net_new');
    
    
    % MSE & R
    
    y_pred = sim(net_new,tst_dat');
    y_pred = y_pred';
    y_pred = max(y_pred,0);

    nn_mse = immse(tst_lbl,y_pred);

    set(0,'DefaultFigureVisible','off');
    [~,~,nn_r] = postreg(tst_lbl',y_pred');
    
    
    %********** Result
    params = [
        i,...
        hiddenLayerSize,...
        epochs,...
        goal,...
        min_grad,...
        max_fail,...
        mu,...
        sigma,...
        lambda,...
        trainRatio,...
        valRatio,...
        testRatio,...
        nn_mse,...
        nn_r,...
        ];
    dlmwrite(csv_file_path,params,'delimiter',',','-append');
 
end
end
end
end
end
end
end
end
end