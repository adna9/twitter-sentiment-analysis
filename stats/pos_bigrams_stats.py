from tsvfiles import tsvreader
from tokenizers import twokenize
from postaggers import arktagger
from nltk import bigrams
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

number_of_most_common = 20

#get noah's ark tags for all messages
def arkTags(messages):
    return arktagger.pos_tag_list(messages)

def countTags(l,tags):
    #initialize list with zeros
    n = [0 for x in l]

    #iterate every message
    for t in tags:
        for x in t:
            n[l.index(x)]+=1

    return n

def uniqueTags(t):
    unique_tags = []
    for tag in t:
        unique_tags+=tag

    unique_tags = set(unique_tags)
    return unique_tags

def subList(messages,labels,s):
    sub=[]
    for i in range(0,len(messages)):
        if labels[i]==s:
            sub.append(messages[i])

    return sub

def posBigrams(tags):
    #find bigrams for every message
    b = [bigrams(x) for x in tags]
    b = [list(x) for x in b]

    #caclulate number of bigrams in every message
    c = [Counter(x) for x in b]

    #do not keep number of appearances of bigram
    #we need to know only if the bigram appears in message
    for counter in c:
        for key in counter.keys():
            if counter.get(key,0)>1:
                counter[key]=1

    #calculate total number of appearances bigrams for all messages
    t = c[0]
    for i in range(1,len(c)):
        t += c[i]

    #find most common bigrams for the messages
    return t

def plotBigrams(a,b,ta,ca,la,tb,cb,lb):
    keys = []
    for x in a.most_common(number_of_most_common):
        keys.append(x[0])

    for x in b.most_common(number_of_most_common):
        keys.append(x[0])

    keys = list(set(keys))

    values_a = [a.get(x,0) for x in keys]
    values_b = [b.get(x,0) for x in keys]

    #plot
    X=np.arange(len(keys))
    plt.bar(X+0.00,[x/float(la)*100 for x in values_a],color=ca,width=0.35,label=ta)
    plt.bar(X+0.25,[x/float(lb)*100 for x in values_b],color=cb,width=0.35,label=tb)

    plt.xticks([x+0.25 for x in range(len(keys))],([x for x in keys]),size=13)
    plt.ylabel("%")
    plt.legend(loc="upper right")
    plt.title("Most common POS bigrams")
    
    plt.show()

#read labels and messages from dataset
dataset = "datasets/tweets#2015.tsv"
#dataset = "datasets/training-set-sample.tsv"
labels, messages = tsvreader.opentsv(dataset)
neutral_messages = subList(messages,labels,"neutral")
positive_messages = subList(messages,labels,"positive")
negative_messages = subList(messages,labels,"negative")
subjective_messages = positive_messages + negative_messages

#noah's ark pos tags
ark_tags = arkTags(messages)
ark_tags_neutral = arkTags(neutral_messages)
ark_tags_positive = arkTags(positive_messages)
ark_tags_negative = arkTags(negative_messages)
ark_tags_subjective = arkTags(subjective_messages)

#calculate and plot most common pos bigrams
bigrams_all = posBigrams(ark_tags)
bigrams_neutral = posBigrams(ark_tags_neutral)
bigrams_positive = posBigrams(ark_tags_positive)
bigrams_negative = posBigrams(ark_tags_negative)
bigrams_subjective = posBigrams(ark_tags_subjective)

#plot bigrams
plotBigrams(bigrams_neutral,bigrams_subjective,"Objective messages","c",len(ark_tags_neutral),"Subjective messages","r",len(ark_tags_subjective))
plotBigrams(bigrams_positive,bigrams_negative,"Positive messages","g",len(ark_tags_positive),"Negative messages","r",len(ark_tags_negative))

