from tsvfiles import tsvreader
from features_subjectivity import features
from tokenizers import twokenize
from postaggers import arktagger
from lexicons import Slang
from evaluation import measures
from classifiers import LogisticRegression, SVM, MajorityClassifier
import matplotlib.pyplot as plt
import numpy as np
from nltk import bigrams
from lexicons import SocalLexicon,MinqingHuLexicon,afinn,NRCLexicon,MPQALexicon,SentiWordNetLexicon
from lexicons.afinn import Afinn
import time

def tokenize(l):
    tokens=[]

    for item in l:
        tokens.append(twokenize.simpleTokenize(item))

    return tokens

def plot_learning_curve(length,features_train,labels_train,features_test,labels_test,title):
    #run for every 10% of training set and compute training error and testing error
    step = length/10
    train_error = []
    test_error = []
    maj_clas_train_error = []
    maj_clas_test_error = []

    for i in range(0,10):
        #train for (i+1)*10 percent of training set
        f = features_train[0:((i+1)*(step))]
        l=labels_train[0:((i+1)*(step))]

        #train classifier for the specific subset of training set
        model = LogisticRegression.train(f,l)

        #get training error
        prediction = LogisticRegression.predict(f,model)
        train_error.append(measures.error(l,prediction))

        #get testing error
        prediction = LogisticRegression.predict(features_test,model)
        test_error.append(measures.error(labels_test,prediction))

        #get testing error for majority classifier
        prediction = MajorityClassifier.predictSubj(features_test)
        maj_clas_test_error.append(measures.error(labels_test,prediction))

        #get training error for majority classifier
        prediction = MajorityClassifier.predictSubj(f)
        maj_clas_train_error.append(measures.error(l,prediction))


    #insert bias
    train_error.insert(0,0)
    test_error.insert(0,1)
    maj_clas_test_error.insert(0,maj_clas_test_error[0])
    maj_clas_train_error.insert(0,1)
    
    x = np.arange(len(train_error))*10
    plt.plot(x,train_error,label="Training Error")
    plt.plot(x,test_error,label="Testing Error")
    plt.plot(x,maj_clas_test_error,label="Majority Classifier - Testing Error")
    plt.plot(x,maj_clas_train_error,label="Majority Classifier - Training Error")
    plt.ylabel('error')
    plt.xlabel("% of messages")
    plt.title(title)
    plt.legend()
    plt.show()

#calculate bigrams of every item of the list l
def getBigrams(l):
    b = []
    for x in l:
        b.append(list(bigrams(x)))

    return b

#def main():

start_time = time.time()

#load training set
dataset_train = "datasets/training-set-sample.tsv"
#dataset_train = "datasets/train15.tsv"
labels_train, messages_train = tsvreader.opentsv(dataset_train)

#load testing set
#dataset_test = "datasets/dev15.tsv"
dataset_test = "datasets/testing-set-sample.tsv"
labels_test, messages_test = tsvreader.opentsv(dataset_test)

#load Slang Dictionary
slangDictionary = Slang.Slang()

#merge positive-negative categories into one category(subjective), as we
#want to check the subjectivity of message
# 0 - objective(neutral) messages
# 1 - subjective(positive or negatve) messages
labels_train = [0 if x=="neutral" else 1 for x in labels_train]
labels_test = [0 if x=="neutral" else 1 for x in labels_test]

#tokenize all messages
tokens_train = tokenize(messages_train)
tokens_test = tokenize(messages_test)

#compute pos tags for all messages
pos_tags_train = arktagger.pos_tag_list(messages_train)
pos_tags_test = arktagger.pos_tag_list(messages_test)

#compute pos tag bigrams for all messages
pos_bigrams_train = getBigrams(pos_tags_train)
pos_bigrams_test = getBigrams(pos_tags_test)

#compoute pos tag bigrams Scores
#TODO

#Load Lexicons

#Socal Lexicon
socal = SocalLexicon.SocalLexicon()
#Minqing Hu Lexicon
minqinghu = MinqingHuLexicon.MinqingHuLexicon()
#Afinn Lexicon
afinn = Afinn()
#NRC Lexicon - 5 different versions
nrc1 = NRCLexicon.NRCLexicon(0)
nrc2 = NRCLexicon.NRCLexicon(1)
nrc3 = NRCLexicon.NRCLexicon(2)
nrc4 = NRCLexicon.NRCLexicon(3)
nrc5 = NRCLexicon.NRCLexicon(4)
#MPQA Lexicon
mpqa = MPQALexicon.MPQALexicon()
#SentiWordNet Lexicon
swn = SentiWordNetLexicon.SentiWordNetLexicon()

lexicons = [socal,minqinghu,afinn,nrc1,nrc2,nrc3,nrc4,nrc5,mpqa,swn]

#get features from train messages
features_train = features.getFeatures(messages_train,tokens_train,pos_tags_train,slangDictionary,lexicons)

#get features from test messages 
features_test = features.getFeatures(messages_test,tokens_test,pos_tags_test,slangDictionary,lexicons)

#train classifier and return trained model
model = LogisticRegression.train(features_train,labels_train)
#model = SVM.train(features_train,labels_train)

#predict labels
prediction = LogisticRegression.predict(features_test,model)
#prediction = SVM.predict(features_test,model)

#calculate accuracy
print "Average F1 : " +str(measures.avgF1(labels_test,prediction))
print "Accuracy : " +str(measures.accuracy(labels_test,prediction))
print "F1 Objective : " +str(measures.F1(labels_test,prediction,0))
print "F1 Subjective : " +str(measures.F1(labels_test,prediction,1))
print "Precision Objective: " +str(measures.precision(labels_test,prediction,0))
print "Precision Subjective: " +str(measures.precision(labels_test,prediction,1))
print "Recall Objective : " +str(measures.recall(labels_test,prediction,0))
print "Recall Subjective : " +str(measures.recall(labels_test,prediction,1))


#plot learning curve
#plot_learning_curve(len(messages_train),features_train,labels_train,features_test,labels_test,"Error")

print("--- %s seconds ---" % (time.time() - start_time))

##if __name__ == "__main__":
##    main() 
    

