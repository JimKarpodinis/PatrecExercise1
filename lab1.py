# Import necessary modules

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import lib
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn


# Read training data
filename = os.path.join(os.getcwd() + '\\data', 'train.txt')
X_train = np.genfromtxt(filename)

train_features = X_train[:, 1:]
train_labels = X_train[:, 0]

# Show sample 131
lib.show_sample(train_features, 130)

# Subplot unique labels
lib.plot_digit_samples(train_features, train_labels)

# Find mean at pixel (10, 10) of digit 0
print('Mean at pixel (10, 10) of digit 0 =',
      '%.4f' % lib.digit_mean_at_pixel(train_features, train_labels, 0))

# Find variance at pixel (10, 10) of digit 0
print('Variance at pixel (10, 10) of digit 0 =',
      '%.4f' % lib.digit_variance_at_pixel(train_features, train_labels, 0))

# Find mean and variance at digit 0
# Show the first ten columns
digit_0_mean = lib.digit_mean(train_features, train_labels, 0)
digit_0_var = lib.digit_variance(train_features, train_labels, 0)
print(digit_0_mean[0, :10])
print()
print(digit_0_var[0, :10])

# Imshow mean and variance at digit 0
lib.show_digit_mean(train_features, train_labels, 0)
lib.show_digit_variance(train_features, train_labels, 0)

# Imshow 20 randomly created zeros drawn from the normal distr of 0
fig = plt.figure(20)
for i in range(20): 
    random_digit = np.random.normal(digit_0_mean, digit_0_var ** 0.5)
    fig.add_subplot(4,5,i+1)
    plt.imshow(random_digit.reshape((16,16)), cmap='gray')  

# Calculate mean and variance at every digit
unique_labels = np.unique(train_labels)

mean_per_digit = np.array(
    [lib.digit_mean(train_features, train_labels, digit).reshape(256,)
        for digit in unique_labels])

var_per_digit = np.array([
    lib.digit_variance(train_features, train_labels, digit).reshape(256,)
    for digit in unique_labels])


# Show the first five columns for every digit
print(mean_per_digit[:, :5])
print(var_per_digit[:, :5])

# Imshow randomly created digits drawn from a normal distr

# Draw samples from the normal dist of each digit
random_samples = np.random.normal(mean_per_digit, var_per_digit**0.5)

# Imshow the samples
fig = plt.figure(10)
for i in range(len(random_samples)):
    fig.add_subplot(2,5,i+1)
    plt.imshow(random_samples[i].reshape((16,16)), cmap='gray')


# Imshow digit mean for every digit
lib.plot_digit_means(train_features, train_labels)

# Read test data
filename = os.path.join(os.getcwd() + '\\data', 'test.txt')
X_test = np.genfromtxt(filename)

test_features = X_test[:, 1:]
test_labels = X_test[:, 0]

# Classify test sample 101
classifier_results = lib.euclidean_distance_classifier(
    test_features[101], mean_per_digit)
print(classifier_results)

# Compare prediction and true value
print(classifier_results == test_labels[101])

# Classify all test samples
classifier_results = (
    lib.euclidean_distance_classifier(test_features, mean_per_digit))

# Calculate accuracy score
print(sum(classifier_results == test_labels)/len(test_labels))

# Implement Euclid scikit-learn estimator
edc = lib.EuclideanDistanceClassifier()

# Fit model
edc.fit(train_features, train_labels)

# Predict digits
y_predict = edc.predict(test_features)

# Calculate accuracy score
print('Accuracy score euclidean distance classifier =',
      edc.score(test_features, test_labels))

# Calculate accuracy using 5-fold cross validation

print('Euclidean Distance Classifier cross validation score=',
      lib.evaluate_classifier(lib.EuclideanDistanceClassifier(),
                              train_features, train_labels))

# Plot decision boundary

# Scale data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(train_features)

# Calculate principal components
pca = PCA(n_components=2)
pca.fit(features_scaled)

unique_labels = np.arange(10)
lib.plot_clf(edc, test_features, test_labels, unique_labels, pca)

# Plot learning curve

train_sizes, train_scores, test_scores = learning_curve(
    lib.EuclideanDistanceClassifier(), train_features, train_labels, cv=5,
    n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5))

lib.plot_learning_curve(train_scores, test_scores, train_sizes)

# Implement Naive Bayes Classifier


# Evaluate custom classifier with computed mean and var
print('Custom Naive Bayes cross validation score =',
      lib.evaluate_custom_nb_classifier(train_features, train_labels))

# Evaluate generalized score
nbc = lib.CustomNBClassifier()
nbc.fit(train_features, train_labels)

print('Generalized error for custom naive bayes =',
      1 - nbc.score(test_features, test_labels))

# Evaluate custom classifier with computed mean and unit var
print('Custom Naive Bayes with unit var cross validation score =',
      lib.evaluate_custom_nb_classifier(
          train_features, train_labels, unit_var=True))

# Evaluate generalized score
nbc = lib.CustomNBClassifier(use_unit_variance=True)
nbc.fit(train_features, train_labels)

print('Generalized error for custom naive bayes with unit var=',
      1 - nbc.score(test_features, test_labels))

# Evaluate sklearn classifier
print('Sklearn Naive Bayes cross validation score = ',
      lib.evaluate_sklearn_nb_classifier(train_features, train_labels))

# Evaluate generalized score
gnb = GaussianNB()
gnb.fit(train_features, train_labels)

print('Generalized error for sklearn naive bayes=',
      1 - gnb.score(test_features, test_labels))

# Compare different classifier scores

# K-Nearest Neighbors
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_features, train_labels)

print('K Nearest Neighbors score =',
      neigh.score(test_features, test_labels))

# Support Vector Machine with linear kernel
lsvc = SVC(kernel='linear')
lsvc.fit(train_features, train_labels)

print('Linear Support Vector Machine score =',
      lsvc.score(test_features, test_labels))

# Support Vector Machine with polynomial kernel
psvc = SVC(kernel='poly')
psvc.fit(train_features, train_labels)

print('Polynomial Support Vector Machine score =',
      psvc.score(test_features, test_labels))

# Support Vector Machine with sigmoid kernel
sigsvc = SVC(kernel='sigmoid')
sigsvc.fit(train_features, train_labels)

print('Sigmoid Support Vector Machine score =',
      sigsvc.score(test_features, test_labels))

# Support Vector Machine with rbf kernel
rbfsvc = SVC(kernel='rbf')
rbfsvc.fit(train_features, train_labels)

print('Rbf Support Vector Machine score =',
      rbfsvc.score(test_features, test_labels))

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(train_features, train_labels)

print('Gaussian Naive Bayes score =',
      gnb.score(test_features, test_labels))

# Euclidean Distance Classifier
ecd = lib.EuclideanDistanceClassifier()
ecd.fit(train_features, train_labels)

print('Euclidean Distance Classifier score=',
      ecd.score(test_features, test_labels))

# Find best VotingClassifier from clfs with complementary mistakes  
best_voting_clf = lib.complementary_clfs(train_features, train_labels)


# Find Best Bagging Classifier
best_bagging_clf = lib.bagging_clf(train_features, train_labels)



# Split training data to training, evaluation data
train_features, eval_features, train_labels, eval_labels = \
      train_test_split(train_features, train_labels, train_size=0.33)

# Use a Data Loader to separate data to batches

train_data = lib.DigitsData(train_features, train_labels, trans=lib.ToTensor())
train_dl = DataLoader(train_data, batch_size = 32, shuffle=True)

eval_data =  lib.DigitsData(eval_features, eval_labels, trans=lib.ToTensor())
eval_dl =  DataLoader(eval_data, batch_size = 32, shuffle=True)

test_data = lib.DigitsData(test_features, test_labels, trans=lib.ToTensor())
test_dl = DataLoader(test_data, batch_size = 32, shuffle=True)

# Find best nn architecture:
lib.best_nn_architecture(train_dl, eval_dl)

# Implement nn as an sklearn estimator

# Create an instance
pnn = lib.PytorchNNModel(lib.myNet(256, 200, 10), nn.CrossEntropyLoss(), torch.optim.SGD(lib.myNet(256, 200, 10).parameters(), 
    lr = 0.002))

# fit the model
pnn.fit(train_features, train_labels)