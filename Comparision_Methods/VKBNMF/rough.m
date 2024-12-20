%mat = rand(692,128);
%similarities = cosineSimilarity(mat,mat);
%writematrix(similarities,'Dataset/gene_sim.txt')
mat = rand(692,692);
for i = 1:692
    mat(i,i)=0;
end
writematrix(mat,'Dataset/gene_sim.txt')
