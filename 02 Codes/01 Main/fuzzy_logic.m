clear variables;
close all;
clc;

%% fuzzy

font_size = 12;
path = './Bitumen/Results/Fuzzy Logic/';
data_path = './Bitumen/Data/trn_tst/';

radii_range = 0.1:0.001:1;

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

%% Fuzzy

i = 0;
for radii = radii_range
    
    i = i + 1;
    % Generate an FIS using subtractive clustering, and specify the cluster center range of influence.
    fis = genfis2(X_trn,y_trn,radii);
    % Evaluate generated FIS fuzzy inference
    y_pred = evalfis(X_tst,fis);
    
    mse = immse(y_tst,y_pred);
    
%     figure;
    [~,~,r] = postreg(y_tst',y_pred');
    xlabel(strcat('Predicted Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
    ylabel(strcat('Measured Bitumen (mgHC/g rock)'),'fontweight','bold','fontsize',font_size);
    title(sprintf('Clustering Radius = %1.3f \n MSE = %.5f, R = %.5f',radii,mse,r),'fontweight','bold','fontsize',font_size);
    saveas(gcf,strcat(path,'fl_toc_ClustRadi_',num2str(radii,'%.3f'), '.png'))
    R(i).r = r;
    MSE(i).mse = mse;
    % Put results in a structure
    FuzzyResult(i).Radius = radii;
    FuzzyResult(i).Fuzzy = fis;
    FuzzyResult(i).r = r;
    FuzzyResult(i).y_pred = y_pred;
    FuzzyResult(i).mse = mse;
    


end

[mse_min_y,idx] = min(cell2mat({FuzzyResult.mse}));
mse_min_x = FuzzyResult(1,idx).Radius;
mse_min_model = FuzzyResult(1,idx).Fuzzy;

save(strcat(path,'FuzzyResult.mat'),'FuzzyResult');
save(strcat(path,'fl_model_mse_min.mat'),'mse_min_model');

new_fontSize = 17;
figure;
hold on
plot(cell2mat({FuzzyResult.Radius}),cell2mat({FuzzyResult.mse}),'LineWidth',2);
plot(mse_min_x,mse_min_y,'o','LineWidth',2,'MarkerSize',8,'MarkerEdgeColor','b');
legend('Bitumen','Location','east');
xlabel('Clustering Radius','fontweight','bold','fontsize',new_fontSize);
ylabel('MSE','fontweight','bold','fontsize',new_fontSize);
ax = gca;
ax.FontSize = new_fontSize;

saveas(gcf,strcat(path,'fl_Plot_r.png'))