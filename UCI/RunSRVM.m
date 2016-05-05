%%%%%%%%%%%%%%%%
%% SPARSE RVM %%
%%%%%%%%%%%%%%%%



for set=[0,1,2,3,4,5,6,7,8,9]
    sf = sprintf('UCI/Fold%d',set);
    X_tr = readNPY(sprintf('%s/X_tr.npy',sf));
    y_tr = readNPY(sprintf('%s/y_tr.npy',sf));
    tic;
    opts = rvmopts(struct('sig',X_tr,'tt',y_tr,'PPhi',X_tr,'chatty',0,'plot',0));
    stats = rvm(opts);
    time = toc;
    beta = stats.mmu_M;
    gamma = double(stats.sel);
    fname = sprintf('%s/sRVM.mat',sf);
    save(fname,'beta','gamma','time','stats');

end