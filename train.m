current_folder = pwd;
ds_path = current_folder + "\Spectrograms";
net = googlenet;

process = process();

[trainDs,valDs,testDs] = process.process_datastore(ds_path);

lgraph = process.modify_layers(net);

options = trainingOptions("adam", ...
    "Plots","training-progress", ...
    "ValidationData",valDs,...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropPeriod",20,...
    "MaxEpochs",40, ...
    "MiniBatchSize",64);

instrument_net = trainNetwork(trainDs,lgraph,options);

save instrument_net instrument_net
