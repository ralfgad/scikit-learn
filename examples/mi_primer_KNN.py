"""
==============================================
Face completion with a multi-output estimators
==============================================

This example shows the use of multi-output estimator to complete images.
The goal is to predict the lower half of a face given its upper half.

The first column of images shows true faces. The next columns illustrate
how extremely randomized trees, k nearest neighbors, linear
regression and ridge regression complete the lower half of those faces.

"""
#print(__doc__)

import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as sio

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

def loo_knn(conjuntos, veces):
    entradasI=conjuntos[0].transpose()
    salidasT=conjuntos[3].transpose()
    loo = LeaveOneOut()
    idea=0
    j=0;
    for train, test in loo.split(entradasI):
        #print("%s %s" % (train, test))
        X_train, X_test, y_train, y_test = entradasI[train], entradasI[test], salidasT[train], salidasT[test]
        knn = KNeighborsRegressor(1, weights='uniform')
        clf = knn.fit(X_train, y_train)
        y_ = clf.predict(X_test)
        idea+=mean_squared_error(y_test,y_)
        j+=1;
    #    print(idea,j,y_test ,y_)
        if j==veces:
            break
    resultado=idea/(j)    
    return resultado
    
if __name__ == '__main__':
    # Load the faces datasets
    datos_matlab = sio.loadmat('./idea.mat')
    training=datos_matlab.get('training1_inicial')
    conjunto=training[0,0]   
    veamos=loo_knn(conjunto,2)
    print(veamos)