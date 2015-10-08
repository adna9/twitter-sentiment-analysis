from tsvfiles import tsvreader
from nltk import word_tokenize,pos_tag
from tokenizers import twokenize
from postaggers import arktagger
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np

#get nltk tags for all messages
def nltkTags(messages):
    data = []
    for message in messages:

        #tokenization with ntlk tokenizer
        #tokens = word_tokenize(message)

        #tokenization with Noah's Ark Simple tokenizer
        tokens = twokenize.simpleTokenize(message)
        tags = pos_tag(tokens)
        tags = [x[1] for x in tags]

        data.append(tags)

    return data

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

def plotPie(n,l,t):
    slices = n
    pie_labels=[x+" - "+str("{:2.2f}".format(slices[l.index(x)]*100/float(sum(n))))+"%" for x in l]
    patches, texts = plt.pie(slices,startangle=90)
    plt.title(t)
    plt.legend(patches,pie_labels,loc="best")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def subList(messages,labels,s):
    sub=[]
    for i in range(0,len(messages)):
        if labels[i]==s:
            sub.append(messages[i])

    return sub 

#main

#read labels and messages from dataset
dataset = "datasets/tweets#2013.tsv"
labels, messages = tsvreader.opentsv(dataset)
neutral_messages = subList(messages,labels,"neutral")
positive_messages = subList(messages,labels,"positive")
negative_messages = subList(messages,labels,"negative")
subjective_messages = positive_messages + negative_messages

#nltk pos tags
#nltk_tags = nltkTags(messages)

#noah's ark pos tags
ark_tags = arkTags(messages)
ark_tags_neutral = arkTags(neutral_messages)
ark_tags_positive = arkTags(positive_messages)
ark_tags_negative = arkTags(negative_messages)
ark_tags_subjective = arkTags(subjective_messages)

#unique nltk tags
#unique_nltk_tags = uniqueTags(nltk_tags)

#unique ark tags
unique_ark_tags = uniqueTags(ark_tags)
unique_ark_tags_neutral = uniqueTags(ark_tags_neutral)
unique_ark_tags_positive = uniqueTags(ark_tags_positive)
unique_ark_tags_negative = uniqueTags(ark_tags_negative)
unique_ark_tags_subjective = uniqueTags(ark_tags_subjective)

#number of tags
#pylab.ylabel("Parts of speech")
#plt.bar(range(2),[len(unique_nltk_tags),len(unique_ark_tags)],align="center")
#plt.xticks(range(2),["Nltk ("+str(len(unique_nltk_tags))+")","Noah's Ark ("+str(len(unique_ark_tags))+")"])
#plt.show()

#find the number of each tag in messages
number_of_tags = countTags(list(unique_ark_tags),ark_tags)
number_of_tags_neutral = countTags(list(unique_ark_tags_neutral),ark_tags_neutral)
number_of_tags_positive = countTags(list(unique_ark_tags_positive),ark_tags_positive)
number_of_tags_negative = countTags(list(unique_ark_tags_negative),ark_tags_negative)
number_of_tags_subjective = countTags(list(unique_ark_tags_subjective),ark_tags_subjective)

#plot results

#plot pies
plotPie(number_of_tags,list(unique_ark_tags),"POS Tags - All messages")
plotPie(number_of_tags_neutral,list(unique_ark_tags_neutral),"POS Tags - Objective messages")
plotPie(number_of_tags_positive,list(unique_ark_tags_positive),"POS Tags - Positive messages")
plotPie(number_of_tags_negative,list(unique_ark_tags_negative),"POS Tags - Negative messages")
plotPie(number_of_tags_subjective,list(unique_ark_tags_subjective),"POS Tags - Subjective messages")


#plot bars - Objective vs Subjective
labels = list(unique_ark_tags)
X=np.arange(len(labels))
#plt.bar(X+0.00,number_of_tags_neutral,color="c",width=0.35,label="Objective messages")
#plt.bar(X+0.25,number_of_tags_subjective,color="r",width=0.35,label="Subjective messages")
plt.bar(X+0.00,[x/float(sum(number_of_tags_neutral))*100 for x in number_of_tags_neutral],color="c",width=0.35,label="Objective messages")
plt.bar(X+0.25,[x/float(sum(number_of_tags_subjective))*100 for x in number_of_tags_subjective],color="r",width=0.35,label="Subjective messages")

plt.ylabel("%")
plt.xticks([x+0.25 for x in range(len(labels))],labels)
plt.legend(loc="upper right")
plt.title("POS Tags")
plt.show()

#plot bars - Positive vs Negative
labels = list(unique_ark_tags)
X=np.arange(len(labels))
#plt.bar(X+0.00,number_of_tags_positive,color="g",width=0.35,label="Positive messages")
#plt.bar(X+0.25,number_of_tags_negative,color="r",width=0.35,label="Negative messages")
plt.bar(X+0.00,[x/float(sum(number_of_tags_positive))*100 for x in number_of_tags_positive],color="g",width=0.35,label="Positive messages")
plt.bar(X+0.25,[x/float(sum(number_of_tags_negative))*100 for x in number_of_tags_negative],color="r",width=0.35,label="Negative messages")

plt.ylabel("%")
plt.xticks([x+0.25 for x in range(len(labels))],labels)
plt.legend(loc="upper right")
plt.title("POS Tags")
plt.show()
