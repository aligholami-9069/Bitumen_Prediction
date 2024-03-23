clear variables;
close all;
clc;

path = './Bitumen/Results/Correlation/';
%% Load Data

col_names = {...
    'CGR'          ,'(GAPI)';...
    'Bitumen'      ,'(mgHC/g rock)';...
    };
font_size = 14;

data = readtable('./Bitumen/Data/Whole data.xlsx', 'Range','A:H');
dataset = data (:,[4 6]);
i = 1:numel(col_names(:,1))-1; 
j = j + 1;
%% 
x_well_a = table2array(dataset ([1:23],[1]));
y_well_a = table2array(dataset ([1:23],[2]));
ycolname_tmp = char(col_names(numel(col_names(:,1)),1));

ycolname = strcat(ycolname_tmp,{' '},col_names(numel(col_names(:,1)),2));
xcolname = char(col_names(i,1));

scatter(x_well_a,y_well_a,"filled",'blue');
xlabel(strcat(xcolname,{' '},col_names(i,2)),'fontweight','bold','fontsize',font_size);
ylabel(ycolname,'fontweight','bold','fontsize',font_size);
r_p = corrcoef(x_well_a,y_well_a);

    title({
        ['Well: A']
        ['R(Pearson) = ' num2str(r_p(2),'%.5f')]
        },'fontweight','bold','fontsize',font_size);
ls = lsline;
set(ls,'color','r')
saveas(gcf,strcat(path,'Figures1',num2str(j),'.png'))
%% 
x_well_b = table2array(dataset ([24:30],[1]));
y_well_b = table2array(dataset ([24:30],[2]));
ycolname_tmp = char(col_names(numel(col_names(:,1)),1));

ycolname = strcat(ycolname_tmp,{' '},col_names(numel(col_names(:,1)),2));
xcolname = char(col_names(i,1));

scatter(x_well_b,y_well_b,"filled",'blue');
xlabel(strcat(xcolname,{' '},col_names(i,2)),'fontweight','bold','fontsize',font_size);
ylabel(ycolname,'fontweight','bold','fontsize',font_size);
r_p = corrcoef(x_well_b,y_well_b);

    title({
        ['Well: B']
        ['R(Pearson) = ' num2str(r_p(2),'%.5f')]
        },'fontweight','bold','fontsize',font_size);
ls = lsline;
set(ls,'color','r')
saveas(gcf,strcat(path,'Figures2',num2str(j),'.png'))
%% 
x_well_c = table2array(dataset ([31:44],[1]));
y_well_c = table2array(dataset ([31:44],[2]));
ycolname_tmp = char(col_names(numel(col_names(:,1)),1));

ycolname = strcat(ycolname_tmp,{' '},col_names(numel(col_names(:,1)),2));
xcolname = char(col_names(i,1));

scatter(x_well_c,y_well_c,"filled",'blue');
xlabel(strcat(xcolname,{' '},col_names(i,2)),'fontweight','bold','fontsize',font_size);
ylabel(ycolname,'fontweight','bold','fontsize',font_size);
r_p = corrcoef(x_well_c,y_well_c);

    title({
        ['Well: C']
        ['R(Pearson) = ' num2str(r_p(2),'%.5f')]
        },'fontweight','bold','fontsize',font_size);
ls = lsline;
set(ls,'color','r')
saveas(gcf,strcat(path,'Figures3',num2str(j),'.png'))
%% 
x_well_d = table2array(dataset ([45:50],[1]));
y_well_d = table2array(dataset ([45:50],[2]));
ycolname_tmp = char(col_names(numel(col_names(:,1)),1));

ycolname = strcat(ycolname_tmp,{' '},col_names(numel(col_names(:,1)),2));
xcolname = char(col_names(i,1));

scatter_mansouri06 = scatter(x_well_d,y_well_d,"filled",'blue');
xlabel(strcat(xcolname,{' '},col_names(i,2)),'fontweight','bold','fontsize',font_size);
ylabel(ycolname,'fontweight','bold','fontsize',font_size);
r_p = corrcoef(x_well_d,y_well_d);

    title({
        ['Well: D']
        ['R(Pearson) = ' num2str(r_p(2),'%.5f')]
        },'fontweight','bold','fontsize',font_size);
ls = lsline;
set(ls,'color','r')
saveas(gcf,strcat(path,'Figures4',num2str(j),'.png'))
%% 
x_well_e = table2array(dataset ([51:58],[1]));
y_well_e = table2array(dataset ([51:58],[2]));
ycolname_tmp = char(col_names(numel(col_names(:,1)),1));

ycolname = strcat(ycolname_tmp,{' '},col_names(numel(col_names(:,1)),2));
xcolname = char(col_names(i,1));

scatter_mansouri57 = scatter(x_well_e,y_well_e,"filled",'blue');
xlabel(strcat(xcolname,{' '},col_names(i,2)),'fontweight','bold','fontsize',font_size);
ylabel(ycolname,'fontweight','bold','fontsize',font_size);
r_p = corrcoef(x_well_e,y_well_e);

    title({
        ['Well: E']
        ['R(Pearson) = ' num2str(r_p(2),'%.5f')]
        },'fontweight','bold','fontsize',font_size);
ls = lsline;
set(ls,'color','r')
saveas(gcf,strcat(path,'Figures5',num2str(j),'.png'))
%% 
x_well_f = table2array(dataset ([59:68],[1]));
y_well_f = table2array(dataset ([59:68],[2]));
ycolname_tmp = char(col_names(numel(col_names(:,1)),1));

ycolname = strcat(ycolname_tmp,{' '},col_names(numel(col_names(:,1)),2));
xcolname = char(col_names(i,1));

scatter_sousangerd03 = scatter(x_well_f,y_well_f,"filled",'blue');
xlabel(strcat(xcolname,{' '},col_names(i,2)),'fontweight','bold','fontsize',font_size);
ylabel(ycolname,'fontweight','bold','fontsize',font_size);
r_p = corrcoef(x_well_f,y_well_f);

    title({
        ['Well: F']
        ['R(Pearson) = ' num2str(r_p(2),'%.5f')]
        },'fontweight','bold','fontsize',font_size);
ls = lsline;
set(ls,'color','r')
saveas(gcf,strcat(path,'Figures6',num2str(j),'.png'))
