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
        train.append(measures.error(l,prediction))

        #get testing error
        prediction = LogisticRegression.predict(features_test,model)
        test.append(measures.error(labels_test,prediction))

        #get error for majority classifier
        prediction = MajorityClassifier.predictPol(features_test)
        maj_clas.append(measures.error(labels_test,prediction))

   
   
    x = np.arange(len(train))*10
    plt.plot(x,train,color="blue",linewidth="2.0",label="Training error")
    plt.plot(x,test,color="blue",linestyle="dashed",linewidth="2.0",label="Testing error")
    plt.plot(x,maj_clas,color="red",linewidth="2.0",label="Majority Classifier error")
    plt.ylim(0,1)
    plt.ylabel('error')
    plt.xlabel("% of messages")
    plt.legend(loc="upper left")
    plt.show()


def plot_recall_precision(length,features_train,labels_train,features_test,labels_test):


    threshold=[0.1 ,0.2 ,0.3 ,0.4,0.5,0.6,0.7,0.8,0.9]
    
    step = length/3
    colors=['b','r','g']
    for i in range(0,3):
        
        #((i+1)*(step)) percent of train data
        f = features_train[0:((i+1)*(step))]
        l=labels_train[0:((i+1)*(step))]

        #train classifier for the specific subset of training set
        model = LogisticRegression.train(f,l)
        
        #recall-precision for every threshold value
        recall = []
        precision=[]

        for t in threshold :

            prediction = LogisticRegression.predict(features_test,model,t)
            
            recall.append(measures.recall(labels_test,prediction,1))
            precision.append(measures.precision(labels_test,prediction,1))

        plt.plot(recall,precision,linewidth="2.0",label=str((i+1)*33)+"% of train data",color=colors[i])

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Positive tweets')
    plt.legend()
    
    plt.show()
         
