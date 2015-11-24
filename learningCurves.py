from evaluation import measures
from classifiers import LogisticRegression, SVM, MajorityClassifier
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_learning_curve(features_train,labels_train,features_test,labels_test,C=1):
    #run for every 10% of training set and compute training error and testing error
    step = len(features_train)/10
   
    train = []
    test = []
    maj_clas = []
   
    for i in range(0,10):
        print i
        
        #train for (i+1)*10 percent of training set
        f = features_train[0:((i+1)*(step))]
        l=labels_train[0:((i+1)*(step))]

        #train classifier for the specific subset of training set
        #model = LogisticRegression.train(f,l)
        model = SVM.train(f,l,c=C,k="linear")

        #get training error
        #prediction = LogisticRegression.predict(f,model)
        prediction = SVM.predict(f,model)
        train.append(measures.accuracy(l,prediction))

        #get testing error
        #prediction = LogisticRegression.predict(features_test,model)
        prediction = SVM.predict(features_test,model)
        test.append(measures.accuracy(labels_test,prediction))

        #get error for majority classifier
        prediction = MajorityClassifier.predictPol(features_test)
        maj_clas.append(measures.accuracy(labels_test,prediction))

   
    #karabatsis = [0.6431]*len(train)
    
    x = np.arange(len(train))*10
    plt.plot(x,train,color="blue",linewidth="2.0",label="Training Error")
    plt.plot(x,test,color="blue",linestyle="dashed",linewidth="2.0",label="Testing Error")
    plt.plot(x,maj_clas,color="red",linewidth="2.0",label="Majority Classifier Error")
    #plt.plot(x,karabatsis,color="green",linewidth="2.0",label="Karabatsis 14")
    plt.ylim(0,1)
    plt.ylabel('Error')
    plt.xlabel("% of messages")
    plt.legend(loc="lower left")
    plt.show()


def plot_recall_precision(length,features_train,labels_train,features_test,labels_test):


    #threshold=[0.1 ,0.2 ,0.3 ,0.4,0.5,0.6,0.7,0.8,0.9]
    threshold = [x / 1000.0 for x in range(0, 1001, 1)]
    
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
            
            recall.append(measures.recall(labels_test,prediction,0))
            precision.append(measures.precision(labels_test,prediction,0))

        plt.plot(recall,precision,linewidth="2.0",label=str((i+1)*33)+"% of train data",color=colors[i])

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Negative tweets')
    plt.legend()
    
    plt.show()

def C_comparison(length,features_train,labels_train,features_test,labels_test):
    C = [0.001,0.05,0.1,0.3,0.5,0.8,1,10,100,350,500,1000,3500,5000,10000,50000,100000]


    scores = []
    for c in C:
        model = LogisticRegression.train(features_train,labels_train,c)

        prediction = LogisticRegression.predict(features_test,model)

        scores.append((measures.avgF1(labels_test,prediction,0,1)))
                      
    plt.plot(C,scores,color="blue",linewidth="2.0")
    plt.xticks(C)
    plt.ylabel("F1")
    plt.xlabel("C")
    plt.show()

def plotFeaturesF1(features_train,labels_train,features_test,labels_test):
    x = list(np.arange(len(features_train[0])))
    #x = list(np.arange(5))
    y = []
    for i in range(0,len(features_train[0])):
            f_train = features_train[:,i]
            f_test = features_test[:,i]
            f_train = f_train.reshape(f_train.shape[0],1)
            f_test = f_test.reshape(f_test.shape[0],1)
            model = LogisticRegression.train(f_train,labels_train)
            prediction = LogisticRegression.predict(f_test,model)
            y.append(measures.avgF1(labels_test,prediction,0,1))
    plt.plot(x,y,color="blue",linewidth="2.0")
    plt.ylabel("F1")
    plt.xlabel("# of Feature")
    plt.xticks(x)
    plt.show() 
         
