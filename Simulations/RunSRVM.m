%%%%%%%%%%%%%%%%
%% SPARSE RVM %%
%%%%%%%%%%%%%%%%

dataFolders = [25,50,75,100,150,200,400];

for set=linspace(0,99,100)
    sf = sprintf('Set%02d',set);
    for df=dataFolders
        X_tr = readNPY(sprintf('%s/n_%03d/X_tr.npy',sf,df));
        y_tr = readNPY(sprintf('%s/n_%03d/y_tr.npy',sf,df));
        tic;
        opts = rvmopts(struct('sig',X_tr,'tt',y_tr,'PPhi',X_tr,'chatty',0,'plot',0));
        stats = rvm(opts);
        time = toc;
        beta = stats.mmu_M;
        gamma = double(stats.sel);
        fname = sprintf('Simulations/%s/n_%03d/sRVM.mat',sf,df);
        save(fname,'beta','gamma','time','stats');
    end
end