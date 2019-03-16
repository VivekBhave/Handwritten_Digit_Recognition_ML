% load training set and testing set
clc;
close all;
clear all;
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

n=1000;
for i=1:n
    trains(i,:)=train_set(i,:);
    tests(i,:)=test_set(i,:);
    trainl(i,:)=train_label(i,:);
    testl(i,:)=train_label(i,:);
end
train_set=trains';
test_set=tests';
train_label=trainl;
test_label=testl;


sumpca = zeros(784,n);

 
for i = 1:n
    sumpca = sumpca + train_set(:,i);
        
end
 
meanpca = sumpca/(n);
 
%scatter matrix 
 scatM=zeros(784,784);
for i = 1:n
    
        scatM = scatM+(train_set(:,i) - meanpca)*(transpose(train_set(:,i) - meanpca));
    
end
 
 
[V,D]=eig(scatM);
 
 
[Dpca ,order] = sort(diag(D),'descend');  %# sort eigenvalues in descending order

dimensions=500;  %change dimensions here
DimRed = V(:,order(1:dimensions));
train_set=(DimRed'*train_set)';
test_set=(DimRed'*test_set)';


% training
tic; 
model = svmtrain(train_label, train_set, '-s 0 -t 0');
t1 = toc;
% classification
tic;
[predicted_label, accuracy, decision_values]=svmpredict(test_label, test_set, model);
t2 = toc;
disp(num2str(t1));
disp(num2str(t2));
