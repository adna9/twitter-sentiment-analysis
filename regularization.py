from sklearn import preprocessing
import numpy as np
import math

#xi=xi-meam/variance
def regularize(features):

    #regularize per column
    for i in range(0,len(features[0])):

        #take evary column
        feat=features[:,i]
    
        #mean and variance of every column
        mean=np.mean(feat)
        var=np.var(feat)

        if(var!=0):
            features[:,i]=(features[:,i]-mean)/float(var)
        else :
            features[:,i]=0
            
     
    return features

#xi=xi-xmin/xman-xmin
def regularizeMaxMin(features):

    #regularize per column
    for i in range(0,len(features[0])):

        #take evary column
        feat=features[:,i]
    
        #max and min value of every feature 
        xmax=max(feat)
        xmin=min(feat)

        if((xmax-xmin)!=0):
            features[:,i]=(features[:,i]-xmin)/float(xmax-xmin)
        else :
            features[:,i]=0
            
     
    return features

