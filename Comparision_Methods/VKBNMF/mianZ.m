clc
clear 
get_data()


%%%%% VKBNMF
%CVa
tic
disp('Start VKBNMF:')
result = {'','AUPR','AUC','f1'};
global coeff;
for data_num = 1:1
    jieguo = CV_VKBNMF_ARD(1,data_num);
    result = [result;[['Data',num2str(data_num)],num2cell(jieguo)]];
end
writecell(result,'VKBNMF_result.xlsx','Sheet','Pairwise_interaction')

%CVr
disp('Start VKBNMF:')
% result = {'','AUPR','AUC','f1'};
result = {'','AUPR','AUC','f1'};
for data_num = 1:1%4
    jieguo = CV_VKBNMF_ARD(2,data_num);
    result = [result;[['Data',num2str(data_num)],num2cell(jieguo)]];
end
writecell(result,'VKBNMF_result.xlsx','Sheet','Human_Protein')

%CVl
disp('Start VKBNMF:')
% result = {'','AUPR','AUC','f1'};
result = {'','AUPR','AUC','f1'};
for data_num = 1:1%4
    jieguo = CV_VKBNMF_ARD(3,data_num);
    result = [result;[['Data',num2str(data_num)],num2cell(jieguo)]];
end

writecell(result,'VKBNMF_result_gene.xlsx','Sheet','Viral_Protein')
toc