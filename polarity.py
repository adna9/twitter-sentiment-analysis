from tsvfiles import tsvreader
from features_subjectivity import features
from tokenizers import twokenize
from postaggers import arktagger
from lexicons import Slang
from classifiers import LogisticRegression,SVM 

def accuracy(A,B):
    #if A[i]==B[i] then A[i]+B[i]==0 or A[i]+B[i]==2
    #else if A[i]/=B[i] the A[i]+B[i]=1
    errors = sum((A+B)==1)

    return (len(A)-errors)/float(len(A))*100

def tokenize(l):
    tokens=[]

    for item in l:
        tokens.append(twokenize.simpleTokenize(item))

    return tokens   

#dataset 
dataset = "datasets/tweets#2013.tsv"

#open dateset
labels, messages =  tsvreader.opentsvPolarity(dataset)


# 0 - negative messages
# 1 - positive messages
labels = [0 if x=="negative" else 1 for x in labels]

#split labels in train-test 
labels_train = labels[:2400]
labels_test = labels[2400:]

#split messages in train-set
messages_train =messages[:2400]
messages_test=messages[2400:]


#load Slang Dictionary
slangDictionary = Slang.Slang()

#tokenize all messages
tokens_train = tokenize(messages_train)
tokens_test = tokenize(messages_test)

#compute pos tags for all messages
pos_tags_train = arktagger.pos_tag_list(messages_train)
pos_tags_test = arktagger.pos_tag_list(messages_test)


#get features from train messages
features_train = features.getFeatures(messages_train,tokens_train,pos_tags_train,slangDictionary)

#train classifier and return trained model
#model = LogisticRegression.train(features_train,labels_train)
model = SVM.train(features_train,labels_train)

#get features from test messages 
features_test = features.getFeatures(messages_test,tokens_test,pos_tags_test,slangDictionary)

#predict labels
#prediction = LogisticRegression.predict(features_test,model)
prediction = SVM.predict(features_test,model)

#calculate accuracy
print "Accuracy : " +str(accuracy(labels_test,prediction))+" %"

