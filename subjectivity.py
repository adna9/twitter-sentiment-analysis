from tsvfiles import tsvreader
from features_subjectivity import features
from tokenizers import twokenize
from postaggers import arktagger 
from evaluation import measures
from classifiers import LogisticRegression, SVM, MajorityClassifier
import matplotlib.pyplot as plt
import numpy as np
from nltk import bigrams
from lexicons import negations,Slang,SocalLexicon,MinqingHuLexicon,afinn,NRCLexicon,MPQALexicon,SentiWordNetLexicon
from lexicons.afinn import Afinn
from clusters import Clusters
import learningCurves
import time

def tokenize(l):
    tokens=[]

    for item in l:
        tokens.append(twokenize.simpleTokenize(item))

    return tokens

#calculate bigrams of every item of the list l
def getBigrams(l):
    b = []
    for x in l:
        b.append(list(bigrams(x)))

    return b

def getBigramsSet(pos_bigrams):
    s = set()
    
    for x in pos_bigrams:
        for bigram in x:
            s.add(bigram)


    return list(s)

def posBigramsScore(bigrams,category,pos_tags_bigrams,labels):
    #keep pos tags bigrams of specific category
    bigrams_category = subList(pos_tags_bigrams,labels,category)

    #initialize dictionary
    d = {}

    #calculate score for every bigram
    for bigram in bigrams:
        d[bigram] = score(bigram,category,bigrams_category,pos_tags_bigrams)


    return d


def score(bigram,category,bigrams_category,pos_tags_bigrams):
    #messages of "category" containing "bigram"
    x1 = 0
    for i in range(0,len(bigrams_category)):
        if bigram in bigrams_category[i]:
            x1+=1

    #messages containing "bigram"
    x2 = 0
    for i in range(0,len(pos_tags_bigrams)):
        if bigram in pos_tags_bigrams[i]:
            x2 += 1

    #messages of "category"
    x3 = len(bigrams_category)

    precision = x1/float(x2)
    recall = x1/float(x3)
    

    #return f1 score
    if precision==0 or recall==0:
        return 0
    
    return (2*precision*recall)/float(precision + recall)
    
def subList(pos_tags,labels,c):
    sub=[]
    for i in range(0,len(pos_tags)):
        if labels[i]==c:
            sub.append(pos_tags[i])

    return sub

def getLexiconF1andPrecision(lexicon, messages, labels):
    #initialize dictionaries
    precision_obj = {}
    f1_obj = {}
    precision_sub = {}
    f1_sub = {}

    #get all words from lexicon
    words = lexicon.d.keys()

    #number of messages that are objective
    x1 = len([x for x in labels if x==0])
    #number of messages that are subjective
    x2 = len([x for x in labels if x==1])

    for word in words:
        #number of messages that contain "word" and are objective
        x3 = 0
        #number of messages that contain "word" and are subjective
        x4 = 0
        #number of messages that contain the "word"
        x5 = 0

        for i in range(0,len(messages)):
            if (word in messages[i]):
                x5+=1

                if(labels[i]==0):
                    x3+=1
                else:
                    x4+=1

        #precision
        if x5!=0:
            precision_obj[word] = x3/float(x5)
            precision_sub[word] = x4/float(x5)
        else:
            precision_obj[word] = 0
            precision_sub[word] = 0

        #recall
        if x1==0:
            recall_obj = 0
        else:
            recall_obj = x3/float(x1)
            
        if x2==0:
            recall_sub = 0
        else:
            recall_sub = x4/float(x2)

        #F1
        if (precision_obj[word] + recall_obj)==0:
            f1_obj[word] = 0
        else:
            f1_obj[word] = (2*precision_obj[word]*recall_obj)/float(precision_obj[word] + recall_obj)

        if (precision_sub[word] + recall_sub)==0:
            f1_sub[word] = 0
        else:
            f1_sub[word] = (2*precision_sub[word]*recall_sub)/float(precision_sub[word] + recall_sub)
            

    return precision_obj, f1_obj, precision_sub, f1_sub

#def main():

start_time = time.time()

#load training set
#dataset_train = "datasets/training-set-sample.tsv"
dataset_train = "datasets/train15.tsv"
labels_train, messages_train = tsvreader.opentsv(dataset_train)

#load testing set
dataset_test = "datasets/dev15.tsv"
#dataset_test = "datasets/testing-set-sample.tsv"
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

#get the unique pos bigrams from training set
unique_bigrams = getBigramsSet(pos_bigrams_train)
#calculate pos bigrams score for all categories
#both dictionaries will be used for training and testing (cannot create new for testing because we don't know the labels of the new messages)
pos_bigrams_scores_objective = posBigramsScore(unique_bigrams,0,pos_bigrams_train,labels_train)
pos_bigrams_scores_subjective = posBigramsScore(unique_bigrams,1,pos_bigrams_train,labels_train)

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

#assign a precision and F1 score to each word of a lexicon
lexicon_precision_objective, lexicon_f1_objective, lexicon_precision_subjective, lexicon_f1_subjective  = getLexiconF1andPrecision(mpqa, messages_train, labels_train)

#get negations list
negationList = negations.loadNegations();

#load word clusters
clusters = Clusters.Clusters()

#get features from train messages
features_train = features.getFeatures(messages_train,tokens_train,pos_tags_train,slangDictionary,lexicons,pos_bigrams_train,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,lexicon_precision_objective, lexicon_f1_objective, lexicon_precision_subjective, lexicon_f1_subjective, negationList,clusters)
#get features from test messages 
features_test = features.getFeatures(messages_test,tokens_test,pos_tags_test,slangDictionary,lexicons,pos_bigrams_test,pos_bigrams_scores_objective,pos_bigrams_scores_subjective,lexicon_precision_objective, lexicon_f1_objective, lexicon_precision_subjective, lexicon_f1_subjective, negationList,clusters)

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
#learningCurves.plot_learning_curve(len(messages_train),features_train,labels_train,features_test,labels_test)

print("--- %s seconds ---" % (time.time() - start_time))

##if __name__ == "__main__":
##    main() 
    

