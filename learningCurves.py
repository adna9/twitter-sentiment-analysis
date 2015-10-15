from evaluation import measures
from classifiers import LogisticRegression, SVM, MajorityClassifier
import matplotlib.pyplot as plt
import numpy as np
import math



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
        train.append(measures.precision(l,prediction,1))

        #get testing error
        prediction = LogisticRegression.predict(features_test,model)
        test.append(measures.precision(labels_test,prediction,1))

        #get error for majority classifier
        prediction = MajorityClassifier.predictPol(features_test)
        maj_clas.append(measures.precision(labels_test,prediction,1))

   
   
    x = np.arange(len(train))*10
    plt.plot(x,train,label="Training precision")
    plt.plot(x,test,label="Testing precision")
    plt.plot(x,maj_clas,label="Majority Classifier precision")

    plt.ylabel('Positives precision')
    plt.xlabel("% of messages")
    plt.legend()
    plt.show()

def plot_recall_precision(length,features_train,labels_train,features_test,labels_test):

    #for every 33% of train messages and compute recall-precision curves
    step = length/3

    threshold=[0,0.1 ,0.2 ,0.3 ,0.4,0.5,0.6,0.7,0.8,0.9]
    
    x=np.arange(len(threshold))*10
    
    for i in range(0,3):

        #((i+1)*(step)) percent of train data
        f = features_train[0:((i+1)*(step))]
        l=labels_train[0:((i+1)*(step))]

        #train classifier for the specific subset of training set
        model = LogisticRegression.train(f,l)

        #recall-precision for every threshold value
        recall = []
        baseline_recall=[]
        precision=[]
        baseline_precision=[]
        
        for t in threshold :

            prediction = LogisticRegression.predict(features_test,model,t)
            baseline_prediction = MajorityClassifier.predictPol(features_test)
            
            recall.append(measures.recall(labels_test,prediction,1))
            precision.append(measures.precision(labels_test,prediction,1))
            baseline_recall.append(measures.recall(labels_test,baseline_prediction,1))
            baseline_precision.append(measures.precision(labels_test,baseline_prediction,1))
            
        
        #plot recall-precision  
        plt.plot(x,recall,label="recall "+str((i+1)*33)+"%")
        plt.plot(x,precision,label="precision"+str((i+1)*33)+"%")
        

    plt.plot(x,baseline_recall,label="baseline recall")
    plt.plot(x,baseline_precision,label="baseline precision")
    
    plt.xlabel("threshold%")
    plt.legend()
    plt.show()


   
def plot_error_threshold(length,features_train,labels_train,features_test,labels_test):
    #for every 33% of train messages and compute recall-precision curves
    step = length/3

    threshold=[0,0.1 ,0.2 ,0.3 ,0.4,0.5,0.6,0.7,0.8,0.9]
    
    x=np.arange(len(threshold))*10
    
    for i in range(0,3):

        #((i+1)*(step)) percent of train data
        f = features_train[0:((i+1)*(step))]
        l=labels_train[0:((i+1)*(step))]

        #train classifier for the specific subset of training set
        model = LogisticRegression.train(f,l)

        #error for every threshold value
        error=[]
        maj_clas=[]
        
        for t in threshold :

            prediction = LogisticRegression.predict(features_test,model,t)
            baseline_prediction = MajorityClassifier.predictPol(features_test)
            
            error.append(measures.error(labels_test,prediction))
            maj_clas.append(measures.error(labels_test,prediction))
        
        plt.plot(x,error,label="error"+str((i+1)*33)+"%")

    plt.plot(x,maj_clas,label="baseline error")
    plt.xlabel("threshold%")
    plt.legend()
    plt.show()
  
