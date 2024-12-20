%%%%Extract data from Dataset and generate Matlab data file "exper_data.mat"
function get_data()

human_paac_sim = importdata('Dataset/pd/gene_sim.txt');
virus_paac_sim = importdata('Dataset/pd/gene_sim.txt');
human_pro_ID = readcell('Dataset/pd/gene_ID.xlsx');
virus_pro_ID = readcell('Dataset/pd/gene_ID.xlsx');
Nh = size(human_paac_sim,1);
Nv = size(virus_paac_sim,1);

inter_tensor_exp = zeros(Nh,Nv,1);%4
dis_name = {'Cardiovascular Infections','Dilated Cardiomyopathy','Endocarditis','Viral Myocarditis'};
for i=1:1 %4
    inter_data = readcell('Dataset/pd/gene_gene_interaction.xlsx');% ,'Sheet',dis_name{i} human_virus_interaction.xlsx
    inter_data = cell2mat(inter_data);
    %size(inter_data)
    %from zero to one indexing
    for j = 1:size(inter_data,1)
        inter_data(j,1) = inter_data(j,1)+1;
        inter_data(j,2) = inter_data(j,2)+1;
    end
    Ni = size(inter_data,1);   
    
    %full(sparse(inter_data(:,1),inter_data(:,2),ones(Ni,1),Nh,Nv))
    inter_tensor_exp(:,:,i) = full(sparse(inter_data(:,1),inter_data(:,2),ones(Ni,1),Nh,Nv));
    %size(inter_tensor_exp)
    for j=1:Nh
        for k=1:Nh
            if inter_tensor_exp(j,k,i)==1
                inter_tensor_exp(k,j,i)=1;
            end
        end
    end

    %size(inter_tensor_exp)
end
exper_data.human_pro_ID = human_pro_ID;
exper_data.human_paac_sim = human_paac_sim;
exper_data.virus_pro_ID = virus_pro_ID;
exper_data.virus_paac_sim = virus_paac_sim;
exper_data.inter_tensor_exp = inter_tensor_exp;
save exper_data exper_data;

end