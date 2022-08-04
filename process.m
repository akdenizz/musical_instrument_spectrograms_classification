classdef process
    methods(Static)
        function [trainDs,valDs,testDs] = process_datastore(ds_path)
            imds = imageDatastore(ds_path,"IncludeSubfolders",true,"LabelSource","foldernames");
            [trainImgs,valImgs,testImgs] = splitEachLabel(imds,0.8,0.1,0.1,"randomized");
            trainDs = augmentedImageDatastore([224 224],trainImgs);
            valDs = augmentedImageDatastore([224 224],valImgs);
            testDs = augmentedImageDatastore([224 224],testImgs);
            save testImgs testImgs
            save testDs testDs
        end
        
        function lgraph = modify_layers(net)
            lgraph = layerGraph(net);
            new_fc = fullyConnectedLayer(14,"Name","new_fc");
            new_out = classificationLayer("Name","new_out");
            lgraph = replaceLayer(lgraph,"loss3-classifier",new_fc);
            lgraph = replaceLayer(lgraph,"output",new_out);
        end
    end
end
