PMG_mse = readNPY('UCI/Summary/PMG_mse.npy');
PMS_mse = readNPY('UCI/Summary/PMS_mse.npy');
OLSR_mse = readNPY('UCI/Summary/olsr_mse.npy');
RIDGE_mse = readNPY('UCI/Summary/ridge_mse.npy');
SRVM_mse = readNPY('UCI/Summary/srvm_mse.npy');
LASSO_mse = readNPY('UCI/Summary/lasso_mse.npy');
LARS_mse = readNPY('UCI/Summary/lars_mse.npy');
LARS2_mse = readNPY('UCI/Summary/lars2_mse.npy');
LARS4_mse = readNPY('UCI/Summary/lars4_mse.npy');
LARS10_mse = readNPY('UCI/Summary/lars10_mse.npy');

figure(1)
boxplot([OLSR_mse', RIDGE_mse', LASSO_mse', LARS_mse', SRVM_mse',PMG_mse(5,:)'],'Labels',{'OLSR','RIDGE','LASSO','LARS','SRVM','DPP'})


PMG_sprs = readNPY('UCI/Summary/PMG_sprs.npy');
PMS_sprs = readNPY('UCI/Summary/PMS_sprs.npy');
% OLSR_sprs = readNPY('UCI/Summary/olsr_sprs.npy');
% RIDGE_sprs = readNPY('UCI/Summary/ridge_sprs.npy');
SRVM_sprs = readNPY('UCI/Summary/srvm_sprs.npy');
LASSO_sprs = readNPY('UCI/Summary/lasso_sprs.npy');
LARS_sprs = readNPY('UCI/Summary/lars_sprs.npy');
LARS2_sprs = readNPY('UCI/Summary/lars2_sprs.npy');
LARS4_sprs = readNPY('UCI/Summary/lars4_sprs.npy');
LARS10_sprs = readNPY('UCI/Summary/lars10_sprs.npy');

figure(2)
boxplot([LASSO_sprs', LARS_sprs', SRVM_sprs',PMG_sprs(5,:)'],'Labels',{'LASSO','LARS','SRVM','DPP'})

