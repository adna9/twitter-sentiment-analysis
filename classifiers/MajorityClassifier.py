#Majority Classifier
#always responds 0-neutral
#always responds 1-positive

import numpy as np

#predict labels
def predictSubj(features):
    return np.zeros(len(features))    

def predictPol(features):
    return np.ones(len(features))
