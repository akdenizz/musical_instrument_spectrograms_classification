load testDs.mat testDs
load testImgs.mat testImgs
load instrument_net.mat instrument_net

testPred = classify(instrument_net,testDs);
acc = nnz(testPred == testImgs.Labels)/numel(testImgs.Labels);
disp(acc)

[cmap,clabel] = confusionmat(testImgs.Labels,testPred);
heatmap(clabel,clabel,cmap)