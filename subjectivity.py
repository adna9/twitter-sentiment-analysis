from tsvfiles import tsvreader
from features_subjectivity import features
from tokenizers import twokenize
from postaggers import arktagger
from lexicons import Slang
from evaluation import measures
from classifiers import LogisticRegression,SVM
import matplotlib.pyplot as plt

def tokenize(l):
    tokens=[]

    for item in l:
        tokens.append(twokenize.simpleTokenize(item))

    return tokens

def plot_learning_curve(train_error,test_error):
    x = [0,10,20,30,40,50,60,70,80,90,100]
    plt.plot(x,train_error,label="Training Error")
    plt.plot(x,test_error,label="Testing Error")
    plt.ylabel('error')
    plt.xlabel("% of messages")
    plt.legend()
    plt.show()

#def main():

#load training set
#dataset_train = "datasets/training-set-sample.tsv"
dataset_train = "datasets/train15.tsv"
labels_train, messages_train = tsvreader.opentsv(dataset_train)
##labels,messages  = tsvreader.opentsv(dataset_train)
##labels_train = labels[0:7000]
##messages_train = messages[0:7000]
##
##labels_test = labels[7000:len(labels)]
##messages_test = messages[7000:len(messages)]

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

#get features from train messages
features_train = features.getFeatures(messages_train,tokens_train,pos_tags_train,slangDictionary)

#get features from test messages 
features_test = features.getFeatures(messages_test,tokens_test,pos_tags_test,slangDictionary)

#train classifier and return trained model
#model = LogisticRegression.train(features_train,labels_train)
#model = SVM.train(features_train,labels_train)

#predict labels
#prediction = LogisticRegression.predict(features_test,model)
#prediction = SVM.predict(features_test,model)

#calculate accuracy
#print "Average F1 : " +str(measures.avgF1(labels_test,prediction))

#run for every 10% of training set and compute training error and testing error
step = len(messages_train)/10
train_error = []
test_error = []

for i in range(0,len(messages_train),step):
    if i+step<len(messages_train):
        f = features_train[0:(i+step)]
        l=labels_train[0:(i+step)]
    else:
        f = features_train[0:len(messages_train)]
        l=labels_train[0:len(messages_train)]


    #train classifier for the specific subset of training set
    model = LogisticRegression.train(f,l)


    #get training error
    prediction = LogisticRegression.predict(f,model)
    train_error.append(measures.error(l,prediction))

    #get testing error
    prediction = LogisticRegression.predict(features_test,model)
    test_error.append(measures.error(labels_test,prediction))

#plot learning curve

##if __name__ == "__main__":
##    main() 
    

