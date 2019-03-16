# Handwritten_Digit_Recognition_ML
My Machine Learning Project on Handwritten Digit Recognition

The codes are run on MNIST dataset.

Run the following codes for respective classifiers from the code folder:

SVM Classifier: SVM.m (uses functions loadMNISTImages and loadMNISTlabels, dataset is loaded locally)
SVM Classifier with PCA: SVMPCA.m (uses functions loadMNISTImages and loadMNISTlabels, dataset is loaded locally)
SVM Classifier with LDA: SVMLDA.m (uses functions loadMNISTImages and loadMNISTlabels, dataset is loaded locally)
Single-Layer CNN: OneLayerNN.py (python code, dataset is loaded from the internet)
Multi-Layer CNN: MultilayerNN.py (python code, dataset is loaded from the internet)


loadMNISTImages.m contains loadMNISTImages function (needs to be in the same location as the SVM codes)
loadMNISTLabels.m contains loadMNISTLabels function (needs to be in the same location as the SVM codes)

train-images.idx3-ubyte contains training images
train-labels.idx1-ubyte contains testing labels
t10k-images.idx3-ubyte contains testing images
t10k-labels.idx1-ubyte contains testing labels
