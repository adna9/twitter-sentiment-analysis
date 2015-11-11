from sklearn import preprocessing
import numpy as mp
import math

#xi=xi-meam/3*variance
def regularize(features):

    #regularize per column
    for i in range(0,len(features[0])):

        #take evary column
        feat=features[:,i]
    
        #mean and variance of every column
        mean=mp.mean(feat)
        var=mp.var(feat)

        if(var!=0):
            features[:,i]=(features[:,i]-mean)/float(3*var)
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

