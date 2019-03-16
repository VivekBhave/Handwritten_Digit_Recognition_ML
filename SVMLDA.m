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


sumlda = zeros(784,n);
    avg = zeros(784,n/2);
 
 
for i = 1:n
    sumlda = sumlda + train_set(:,i);
        
end
    

totalmeanlda=sumlda/(n); %total mean
 
for a=1:(n/2)
    mean_vector(:,a)=(train_set(:,a)+train_set(:,a+200))/2; %mean per class
end

sw = zeros(784,784);

for i = 1:(n/2)
    
    si = train_set(:,i) - mean_vector(:,i) ; %Si matrix
   
    
    sw = sw+si;  %Sw matrix
end
 
 
sb = zeros(784,784);
 
for i = 1:(n/2)
    d = mean_vector(:,i)-totalmeanlda;
    sb = sb+2*(d*transpose(d));  %Sb matrix
end


[V,D]=eig(sb,sw);
  
[Dlda ,order] = sort(diag(D),'descend');  %# sort eigenvalues in descending order

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
