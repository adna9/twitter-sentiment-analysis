import polarity
import subjectivity
from tsvfiles import tsvreader
import time
from clusters import Clusters
from lexicons import negations,Slang,SocalLexicon,MinqingHuLexicon,afinn,NRCLexicon,MPQALexicon,SentiWordNetLexicon
from lexicons.afinn import Afinn
import enchant
from utilities import polaritySubList
from preProcess import *
from postaggers import arktagger
from utilities import *

t1 = time.time()

#load training set
dataset_train = "datasets/training-set-sample.tsv"
#dataset_train = "datasets/train15.tsv"
#dataset_train = "datasets/tweets#2013.tsv"
labels_train,messages_train=tsvreader.opentsv(dataset_train)

#load testing set
dataset_test = "datasets/testing-set-sample.tsv"
#dataset_test = "datasets/dev15.tsv"
#dataset_test = "datasets/devtweets2013.tsv"
labels_test, messages_test = tsvreader.opentsv(dataset_test)

#get negations list
negationList = negations.loadNegations();

#load word clusters
clusters = Clusters.Clusters()

#load Slang Dictionary
slangDictionary = Slang.Slang()

#load general purpose dictionary
dictionary = enchant.Dict("en_US")

#tokenize all messages
tokens_train = tokenize(messages_train)
tokens_test = tokenize(messages_test)

#compute pos tags for all messages (after preprocessing the messages pos tags will be recomputed)
pos_tags_train = arktagger.pos_tag_list(messages_train)
pos_tags_test = arktagger.pos_tag_list(messages_test)

#preprocess messages
process_messages_train = preprocessMessages(messages_train,tokens_train,pos_tags_train,slangDictionary,dictionary)
process_messages_test = preprocessMessages(messages_test,tokens_test,pos_tags_test,slangDictionary,dictionary)

#tokenize process messages (final pos tags)
process_tokens_train=tokenize(process_messages_train)
process_tokens_test=tokenize(process_messages_test)

#compute pos tags for all preprocessed messages
pos_tags_train = arktagger.pos_tag_list(process_messages_train)
pos_tags_test = arktagger.pos_tag_list(process_messages_test)

#Load Lexicons

#Minqing Hu Lexicon
minqinghu = MinqingHuLexicon.MinqingHuLexicon()
#Afinn Lexicon
afinn = Afinn()
#NRC Lexicons
nrc2 = NRCLexicon.NRCLexicon(1)
nrc5 = NRCLexicon.NRCLexicon(4)
nrc6 = NRCLexicon.NRCLexicon(5)
#SentiWordNet Lexicon
swn = SentiWordNetLexicon.SentiWordNetLexicon(False)
#SentiWordNet Lexicon - AverageScores
swn_avg= SentiWordNetLexicon.SentiWordNetLexicon(True)

#do not include MPQA Lexicons
lexicons = [minqinghu,afinn,nrc2,nrc5,nrc6,swn,swn_avg]

#MPQA Lexicons (8 Lexicons)
S_pos = MPQALexicon.MPQALexicon(0)
S_neg = MPQALexicon.MPQALexicon(1)
S_pos_neg = MPQALexicon.MPQALexicon(2)
S_neu = MPQALexicon.MPQALexicon(3)
W_pos = MPQALexicon.MPQALexicon(4)
W_neg = MPQALexicon.MPQALexicon(5)
W_pos_neg = MPQALexicon.MPQALexicon(6)
W_neu = MPQALexicon.MPQALexicon(7)

#SEMEVAL_13 Lexicons
semval_neutral = MPQALexicon.MPQALexicon(8)
semval_positive = MPQALexicon.MPQALexicon(9)
semval_negative = MPQALexicon.MPQALexicon(10)

#MPQA + SEMEVAL_13 Lexicons
mpqa_lexicons = [S_pos,S_neg,S_pos_neg,S_neu,W_pos,W_neg,W_pos_neg,W_neu,semval_neutral,semval_positive,semval_negative]

#STAGE1
#predict subjectivity
prediction_subj=subjectivity.classify(messages_train,labels_train,messages_test,process_messages_train,process_messages_test,tokens_train,tokens_test,process_tokens_train,process_tokens_test,pos_tags_train,pos_tags_test,negationList,clusters,slangDictionary,lexicons,mpqa_lexicons)
#evaluate classification
#print "Subjectivity Detection"
#subjectivity.evaluate(prediction_subj,labels_test)

temp1 = []
temp2 = []
temp3 = []
temp4 = []
temp5 = []
temp6 = []

###ignore messages that havent pass stage1 
##for i in range(0,len(prediction)) :
##    if prediction[i]==0 :
##        labels_test[i]=-1

#keep only the testing data that were classified as subjective
for i in range(0,len(prediction_subj)):
    if prediction_subj[i]==1:
        temp1.append(messages_test[i])
        temp2.append(process_messages_test[i])
        temp3.append(tokens_test[i])
        temp4.append(process_tokens_test[i])
        temp5.append(pos_tags_test[i])
        temp6.append(labels_test[i])

messages_test = temp1
process_messages_test = temp2
tokens_test = temp3
process_tokens_test = temp4
pos_tags_test = temp5
labels_test_2 = temp6      

#keep training data without neutrals
process_messages_train = polaritySubList(process_messages_train,labels_train)
tokens_train = polaritySubList(tokens_train,labels_train)
pos_tags_train = polaritySubList(pos_tags_train,labels_train)
messages_train = polaritySubList(messages_train,labels_train)
process_tokens_train = polaritySubList(process_tokens_train,labels_train)
labels_train = polaritySubList(labels_train,labels_train)

#STAGE2 
#predict polarity       
prediction_pol=polarity.classify(messages_train,labels_train,messages_test,process_messages_train,process_messages_test,tokens_train,tokens_test,process_tokens_train,process_tokens_test,pos_tags_train,pos_tags_test,negationList,clusters,slangDictionary,lexicons,mpqa_lexicons)

#final prediction
prediction = []
#prediction = prediction_subj
for i in range(0,len(prediction_subj)):
    prediction.append(prediction_subj[i])
    
i = 0
for j in range(0,len(prediction)):
    if prediction[j]==1:
        if prediction_pol[i]==0:
            prediction[j]=-1
        else:
            prediction[j]=1
        i+=1
  
#evaluate classification
#print "Polarity Detection"
polarity.evaluate(prediction,labels_test)

t2 = time.time()

print "total time : "+str(t2-t1)
