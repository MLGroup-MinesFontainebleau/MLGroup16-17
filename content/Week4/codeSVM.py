"""
SVM - some examples

"""


#PACKAGE
import numpy as np
import matplotlib.pyplot as plt #data visualization
from matplotlib import style    #data visualization
style.use("ggplot")
from sklearn import svm         #package for SVM




####            basic example with linear separable data      #################
###############################################################################

#training set
#############

#input
X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])
#output
y = np.array([0,1,0,1,0,1])
# y=1 : red point ; y=0 : blue point
colors =np.where(y<1,"blue","red")
print(colors)

#visualization
plt.scatter(X[:,0],X[:,1],c=colors)
plt.show()


# SVM #
#######
# svm.SVC : http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# c : penalty parameter (pour python, 1/C apparait dans le programme de minimisation
#devant 1/2 norme(w)² )
# linear kernel : K(x,z) = (x)t z : equivalent to the non-kernel case
clf = svm.SVC(kernel='linear',C=1.0)
#other possible functions : svm.NuSVC and svm.LinearSVC

#other kernel :
#clf = svm.SVC(kernel='poly',C=1.0,ga)
    

# To find w* and b* with the training set
clf.fit(X,y)



##coefficients w and b

#only available for linear kernel
w = clf.coef_[0]
print(w)
a = -w[0] / w[1]

#intercept
b = clf.intercept_
print(b)


##Support vector
# get support vectors
clf.support_vectors_
## get indices of support vectors
clf.support_ 
## get number of support vectors for each class
clf.n_support_ 



#prediction for news inputs
new1= [[0.58,0.76]]
y_new1=clf.predict(new1)

new2= [[5,7]]
y_new2=clf.predict(new2)


#visualization
# hyperplan equation : x_2=-b/w2 - w1 x1/w2
#slope : -w1/w2
a = -w[0] / w[1]
#intercept : -b/w2
intercept = -b/w[1]


#we add the new points to the data
newInouts=np.append(new1,new2, axis=0) #concatenation of the new inputs
X=np.append(X,newInouts, axis=0) #concatenation with initial data
y=np.append(y, [y_new1,y_new2]) # concatenation of the new outputs with initial outputs
colors=np.append(colors, np.where(y[len(y)-2:]<1,"blue","red"))

# hyperplan representation
xx = np.linspace(0,12)
yy = a * xx +intercept
h0 = plt.plot(xx, yy, 'k-')

plt.scatter(X[:, 0], X[:, 1], c = colors)
plt.legend()
plt.show()


####   example with different kernels (non linearly separable data     ########
###############################################################################


# Code source: Gaël Varoquaux
# License: BSD 3 clause


# Our dataset and targets
X = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          (1, 1),
          # --
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
Y = [0] * 8 + [1] * 8

# figure number
fignum = 1

# fit the model
#beginFor
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=2,C=np.inf)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)

    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
    #endFor
plt.show()


##############################################################################
##############################################################################
# example extracted from : http://eric.univ-lyon2.fr/~ricco/cours/slides/svm.pdf
# to download the ionosphere data : https://archive.ics.uci.edu/ml/datasets/Ionosphere

#importation des données
from math import *
import pandas
import sklearn.datasets


#download the data
data = pandas.read_table("C:\\Users\mdemangeot\\Documents\\ionosphere.txt",sep=",",header=0,decimal=".")
print(data.shape) #35 colonnes, la dernière est celle de la variable binaire et 351 individus
pandas.DataFrame.describe(data)


#training data : 75% of data
dtrain = data[:floor(0.65*351)]
#test data : 25%
dtest = data[floor(0.65*351)+1:]

#dtrain = pandas.read_table("ionosphere-train.txt",sep="\t",header=0,decimal=".")
print(dtrain.shape)
y_app = dtrain.as_matrix()[:,34]
X_app = dtrain.as_matrix()[:,0:33]
#importation de la classe de calcul
from sklearn.svm import SVC
svm = SVC(kernel='poly',gamma=3) #instanciation de l’objet
#affichage des paramètres (on choisit ici un noyau gaussien)
#pas de standardisation (scale) des données apparemment
print(svm)
#apprentissage – construction du modèle prédictif
svm.fit(X_app,y_app)
#importation des données test
#dtest = pandas.read_table("ionosphere-test.txt",sep="\t",header=0,decimal=".")
print(dtest.shape)
y_test = dtest.as_matrix()[:,34]
X_test = dtest.as_matrix()[:,0:33]
#prédiction sur l’échantillon test
y_pred = svm.predict(X_test)
#evaluation : taux d'erreur = 0.07
from sklearn import metrics
err = 1.0 - metrics.accuracy_score(y_test,y_pred)
print(err)


#CROSS VALIDATION 

#classe grille de recherche
from sklearn.grid_search import GridSearchCV
#GridSearchCV permet, avec de la cross validation de choisir le vecteur
 #de paramètres qui minimisent ton erreur !
#paramètres à tester – jouer sur les noyaux et le ‘cost parameter’
parametres = {"kernel":['linear','poly','rbf','sigmoid'],"C":[0.1,0.5,1.0,2.0,10.0]}
#classifieur à utiliser
svmc = SVC()
#instanciation de la recherche
grille = GridSearchCV(estimator=svmc,param_grid=parametres,scoring="accuracy")
#lancer l'exploration
resultats = grille.fit(X_app,y_app)
#meilleur paramétrage : {‘kernel’ : ‘rbf’, ‘C’ : 10.0}
print(resultats.best_params_)
#prédiction avec le ‘’meilleur’’ modèle identifié
ypredc = resultats.predict(X_test)
#performances du ‘’meilleur’’ modèle – taux d’erreur = 0.045 (!)
err_best = 1.0 - metrics.accuracy_score(y_test,ypredc)
print(err_best)

##############################################################################
##############################################################################
