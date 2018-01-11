# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from time import time
import pylab as pl
from sklearn.datasets import fetch_lfw_people

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')


iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################

# Q1

X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf_linear = svm.SVC(C=1, kernel="linear")

parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
svr = svm.SVC()
clf_linear = GridSearchCV(svr, parameters)
clf_linear.fit(X_train, y_train)
clf_linear.score(X_test, y_test)

print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))

# Q2

Cs = list(np.logspace(-3, 3, 5))
gammas = 10. ** np.arange(1, 2)
degrees = np.r_[1, 2, 3]
parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}

svr = svm.SVC()
clf_poly = GridSearchCV(svr, parameters)
clf_poly.fit(X_train, y_train)
clf_poly.score(X_test, y_test)
clf_poly.best_params_
print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))

plt.figure(figsize=(12, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
plot_2d(X, y)
frontiere(clf_linear.predict, X, y)
plt.title("linear kernel")

plt.subplot(133)
plot_2d(X, y)
frontiere(clf_poly.predict, X, y)
plt.title("polynomial kernel")
plt.tight_layout()

###############################################################################
#               SVM GUI
###############################################################################

# please open a terminal and run python2 svm_gui.py
# Then, play with the applet : generate various datasets and observe the
# different classifiers you can obtain by varying the kernel


###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data (if not already on disk); load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images / 255.
n_samples, h, w, n_colors = images.shape


# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(np.int)

####################################################################
# Extract features

# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

####################################################################
# Split data into a half training and half test set

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] / 2], indices[X.shape[0] / 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test =\
    images[train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set

# Q5
print "Linear kernel"
print "Fitting the classifier to the training set"
t0 = time()

Cs = 10. ** np.arange(-5, 6)
scores = []
for C in Cs:
    clf = svm.SVC(kernel="linear", C=C)
    clf.fit(X_train, y_train)
    scores += [clf.score(X_test, y_test)]
ind = np.argmax(scores)
print "Best C: {}".format(Cs[ind])
plt.figure()
plt.plot(Cs, scores)
plt.xlabel("Parametres de regularisation C")
plt.ylabel("Scores d'apprentissage")
plt.xscale("log")
plt.tight_layout()
plt.show()
print "Best score: {}".format(np.max(scores))


print "Predicting the people names on the testing set"
t0 = time()

# predict labels for the X_test images
clf = svm.SVC(kernel="linear", C=Cs[np.argmax(scores)])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print "done in %0.3fs" % (time() - t0)
print "Chance level : %s" % max(np.mean(y), 1. - np.mean(y))
print "Accuracy : %s" % clf.score(X_test, y_test)

# TODO: the same for polynomial kernel

####################################################################
# Qualitative evaluation of the predictions using matplotlib

prediction_titles = [title(y_pred[i], y_test[i], names)
                     for i in range(y_pred.shape[0])]

plot_gallery(images_test, prediction_titles)
pl.show()

####################################################################
# Look at the coefficients
pl.figure()
pl.imshow(np.reshape(clf.coef_, (h, w)))
plt.show()



# Q6

def run_svm_cv(X, y):
    
    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = indices[:X.shape[0] / 2], indices[X.shape[0] / 2:]
    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 5))}
    svr = svm.SVC()
    clf_linear = GridSearchCV(svr, parameters)
    clf_linear.fit(X_train, y_train)

    print('Generalization score for linear kernel: %s, %s \n' %
          (clf_linear.score(X_train, y_train), clf_linear.score(X_test, y_test)))

print "Score sans variable de nuisance"
run_svm_cv(X, y)

print "Score avec variable de nuisance"
n_features = X.shape[1]
# On rajoute des variables de nuisances
sigma = 1
noise = sigma * np.random.randn(n_samples, 300, )
X_noisy = np.concatenate((X, noise), axis=1)
X_noisy = X_noisy[np.random.permutation(X.shape[0])]
run_svm_cv(X_noisy, y)


print "Score apres reduction de dimension"
from sklearn.decomposition import PCA

n_components = 10000  # jouer avec ce parametre
pca = PCA(n_components=n_components).fit(X_noisy)
X_noisy_pca = pca.transform(X_noisy)
run_svm_cv(X_noisy_pca, y)



