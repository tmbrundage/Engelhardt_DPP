% folders = {'20160408_111739';'20160408_130007';'20160408_130949';'20160408_131216';'20160408_131421'};
models = {'DPP_PO';'DPP_PO_greedy';'LASSO';'OLSR';'ORACLE';'RIDGE'};
data   = {'mse';'beta'};
formatString = '%f%f%f%f%f%f';
output = cell(length(data),length(models));
for i = 1:length(data)
    for folder = 1:length(folders)
       for model = 1:length(models)
            fn = strcat(folders{folder},'/output/',models{model},'_',data{i},'.txt');
            f = fopen(fn,'r');
            cols = textscan(f,formatString);
            N = length(cols{length(cols)}); % N is length of last column
            
            add = [];
            for j = 1:length(cols)
                add = [add, cols{j}(1:N)];
            end
            output{i,model} = [output{i}; add];
            fclose(f);
       end
    end
end

output
            