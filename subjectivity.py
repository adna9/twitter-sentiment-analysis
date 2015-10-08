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

#def main():

#load training set
#dataset_train = "datasets/training-set-sample.tsv"
dataset_train = "datasets/tweets#2013.tsv"
##labels_train, messages_train = tsvreader.opentsv(dataset_train)

labels,messages  = tsvreader.opentsv(dataset_train)
labels_train = labels[0:10000]
messages_train = messages[0:10000]
labels_test = labels[10000:len(labels)]
messages_test = messages[10000:len(messages)]

#load testing set
##dataset_test = "datasets/testing-set-sample.tsv"
##labels_test, messages_test = tsvreader.opentsv(dataset_test)

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

##if __name__ == "__main__":
##    main() 
    

