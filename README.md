##Recurrent Convolutional Network for Video-based Person Re-Identification

Code for our CVPR 2016 paper that performs video re-identification.

If you use this code please cite:

```
@inproceedings{mclaughlinrecurrent,
  	title={Recurrent Convolutional Network for Video-based Person Re-Identification},
  	author={McLaughlin, N and Martinez del Rincon, J and Miller, P},
  	booktitle={CVPR},
  	year={2016}
}
```

##Summary
We perform video re-identification by taking a sequence of images and training a neural network to produce a single feature that represents the whole sequence. The feature vectors for different sequences are compared using Euclidean distance. The distance matrix is passed through reranking method which produces Jaccard dissimilarity. Final distance is computed as a linear combination of Euclidean distance and Jaccard dissimilarity. A smaller distance indicates increased similarity between sequences. The sequence feature is produced using attention temporal pooling which includes the mutual influence of both the video sequences in the context of matching.

##Information
A slightly cleaned up implementation of our video re-id system is provided here. If possible I will clean-up and improve the code in future.

This code is capable of training a video re-identification network on the iLids video, PRID or MARS datasets and saving the learned network for later use. The saved network parameters can be loaded from disk and used to run the evaluation code without needing to train the network again.

The optical flow files were produced using the Matlab code in computeOpticalFlow.m 

This matlab code should be used to generate optical flow files before training the neural network. Alternatively, use the flag –dissableOpticalFlow

NOTE - Modify lines 70-77 of videoReid.lua to point to the directories containing the video-reid datasets and generated optical flow files

##Running the code

For this code to run you must have Torch7 installed with the nn, nnx, cunn, rnn, image, optim and cutorch pacakges.

You must have an Nvidia GPU in order to use CUDA. See http://torch.ch/ for details.

Example command-line options to retrain the ResNet:
th main.lua -retrain resnet-x.t7 -data [path-to-CNNTrainRes-directory] -resetClassifier true -nClasses [*n]
([*n]=300 for iLids-VID, [*n]=200 for PRID 2011 and [*n]=1261 for MARS)

Example command-line options to extract features using the finetuned Resnet:
th CNN_extract_features.lua [name-of-the-best-model]

There are four variants of the model go to the respective folder.

Example command-line options that will allow you to train the full model and produce the CMC Rank:
th videoReid.lua -nEpochs 500 -dataset 1 -dropoutFrac 0.6 -sampleSeqLength 16 -samplingEpochs 100 -seed 1
