from evaluation import measures
from classifiers import LogisticRegression, SVM, MajorityClassifier
import matplotlib.pyplot as plt
import numpy as np



def plot_learning_curve(length,features_train,labels_train,features_test,labels_test):
    #run for every 10% of training set and compute training error and testing error
    step = length/10

    train = []
    test = []
    maj_clas = []
   
    for i in range(0,10):

        #train for (i+1)*10 percent of training set
        f = features_train[0:((i+1)*(step))]
        l=labels_train[0:((i+1)*(step))]

        #train classifier for the specific subset of training set
        model = LogisticRegression.train(f,l)

        #get training error
        prediction = LogisticRegression.predict(f,model)
        train.append(measures.avgF1(l,prediction))

        #get testing error
        prediction = LogisticRegression.predict(features_test,model)
        test.append(measures.avgF1(labels_test,prediction))

        #get error for majority classifier
        prediction = MajorityClassifier.predictPol(features_test)
        maj_clas.append(measures.avgF1(labels_test,prediction))

        
    #insert bias
    train.insert(0,1)
    test.insert(0,1)
    maj_clas.insert(0,1)
    
    x = np.arange(len(train))*10
    plt.plot(x,train,label="Training avgF1")
    plt.plot(x,test,label="Testing avgF1")
    plt.plot(x,maj_clas,label="Majority Classifier avgF1")

    plt.ylabel('avgF1')
    plt.xlabel("% of messages")
    plt.legend()
    plt.show()
