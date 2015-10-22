from lexicons import SocalLexicon,MinqingHuLexicon,afinn,NRCLexicon,MPQALexicon,SentiWordNetLexicon
from lexicons.afinn import Afinn
from tokenizers import twokenize
from tsvfiles import tsvreader
from postaggers import arktagger
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def sortByCategory(l,s):
    return [list(x) for x in zip(*sorted(zip(l, s), key=itemgetter(0)))][1]

def plotHistogram(l,a,b,c=0,t="No title"):
    #find plot positions on X axis
##    X = np.arange(a,b)
##    Y = np.arange(b,c)
##    Z = np.arange(c,len(l))
##    
##    plt.bar(X,l[a:b],color="r",width=0.05,label="Negative messages")
##    plt.bar(Y,l[b:c],color="c",width=0.05,label="Neutral messages")
##    plt.bar(Z,l[c:len(l)],color="g",width=0.05,label="Positive messages")
##
##    plt.xlabel("# of message")
##    plt.ylabel("sentiment score")
##    plt.title(t)

    if c==0:
        data1 = np.array(l[a:b])
        data2 = np.array(l[b:len(l)])

        plt.hist(data1,normed=True, color="c", histtype='step',label="Objective messages")
        plt.hist(data2,normed=True, color="r", histtype='step',label="Subjective messages")
    else:
        data1 = np.array(l[a:b])
        data2 = np.array(l[b:c])
        data3 = np.array(l[c:len(l)])

        plt.hist(data1,normed=True, color="r", histtype='step',label="Negative messages")
        plt.hist(data2,normed=True, color="c", histtype='step',label="Neutral messages")
        plt.hist(data3,normed=True, color="g", histtype='step',label="Positive messages")

    
    plt.legend(loc="upper right")
    plt.title(t)
    plt.xlabel("sentiment score")
    plt.ylabel("# of messages")

    plt.show()

def plotDistribution(l,a,b,c=0,t="No title"):

    d = []
    if c==0:
        data1 = np.array(l[a:b])
        data2 = np.array(l[b:len(l)])

        d=[(data1,"c","Objective"), (data2,"r","Subjective")]
    else:
        data1 = np.array(l[a:b])
        data2 = np.array(l[b:c])
        data3 = np.array(l[c:len(l)])

        d=[(data1,"r","Negative"), (data2,"c","Objective"), (data3,"g","Positive")]

    for data in d:
        #fit a normal distribution to the data
        mu, std = norm.fit(data[0])
        lb = data[2]+" : mu = %.2f,  std = %.2f" % (mu, std)

        #plot histogram
        plt.hist(data[0], normed=True, alpha=0, color='g')

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, data[1], linewidth=2, label=lb)


    plt.title(t)
    plt.legend(loc="upper right")
    plt.xlabel("sentiment score")

    plt.show()
    
#read labels and messages from dataset
dataset = "datasets/train15.tsv"
#dataset = "datasets/training-set-sample.tsv"
labels, messages = tsvreader.opentsv(dataset)

##labels = labels[0:100]
##messages = messages[0:100]

#pos tags of messages
tags = arktagger.pos_tag_list(messages)

#initialize lists that hold the sentiment score of every message for every Lexicon
socal_scores = []
minqinghu_scores = []
afinn_scores = []
nrc1_scores = []
nrc2_scores = []
nrc3_scores = []
nrc4_scores = []
nrc5_scores = []
mpqa_scores = []
swn_scores = []

#Lexicon objects

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

#compute sentiment score for all messages
for i in range(0,len(messages)):
    print i
    #tokens of message, used in counting sentiment score
    tokens = twokenize.simpleTokenize(messages[i])

    #compute pos tags of message
    pos_tags = tags[i]

    #update scores
    socal_scores.append(socal.score(tokens))
    minqinghu_scores.append(minqinghu.score(tokens))
    afinn_scores.append(afinn.score(messages[i]))   #Afinn : input message instead of message's tokens
    nrc1_scores.append(nrc1.score(tokens))
    nrc2_scores.append(nrc2.score(tokens))
    nrc3_scores.append(nrc3.score(tokens))
    nrc4_scores.append(nrc4.score(tokens))
    nrc5_scores.append(nrc5.score(tokens))
    mpqa_scores.append(mpqa.score(tokens))
    swn_scores.append(swn.score(tokens,pos_tags))   #SentiWordNet : input messages's tokens and pos tags

#keep only 2 categories(objective-subjective)
labels = [x for x in labels if x == "neutral"] + ["subjective" for x in labels if (x=="positive" or x=="negative")]

#sort all lists by category
socal_scores = sortByCategory(labels,socal_scores)
minqinghu_scores = sortByCategory(labels,minqinghu_scores)
afinn_scores = sortByCategory(labels,afinn_scores)
nrc1_scores = sortByCategory(labels,nrc1_scores)
nrc2_scores = sortByCategory(labels,nrc2_scores)
nrc3_scores = sortByCategory(labels,nrc3_scores)
nrc4_scores = sortByCategory(labels,nrc4_scores)
nrc5_scores = sortByCategory(labels,nrc5_scores)
mpqa_scores = sortByCategory(labels,mpqa_scores)
swn_scores = sortByCategory(labels,swn_scores)
labels.sort()

#category indexes
##negative = labels.index("negative")
##neutral = labels.index("neutral")
##positive = labels.index("positive")

neutral = labels.index("neutral")
subjective = labels.index("subjective")

#plot results
##plotHistogram(socal_scores,negative,neutral,positive,"Socal Lexicon")
##plotHistogram(minqinghu_scores,negative,neutral,positive,"Minqing Hu Lexicon")
##plotHistogram(afinn_scores,negative,neutral,positive,"Afinn Lexicon")
##plotHistogram(nrc1_scores,negative,neutral,positive,"NRC v1 Lexicon")
##plotHistogram(nrc2_scores,negative,neutral,positive,"NRC v2 Lexicon")
##plotHistogram(nrc3_scores,negative,neutral,positive,"NRC v3 Lexicon")
##plotHistogram(nrc4_scores,negative,neutral,positive,"NRC v4 Lexicon")
##plotHistogram(nrc5_scores,negative,neutral,positive,"NRC v5 Lexicon")
##plotHistogram(mpqa_scores,negative,neutral,positive,"MPQA Lexicon")
##plotHistogram(swn_scores,negative,neutral,positive,"SentiWordNet Lexicon")

##plotHistogram(socal_scores,neutral,subjective,t="Socal Lexicon")
##plotHistogram(minqinghu_scores,neutral,subjective,t="Minqing Hu Lexicon")
##plotHistogram(afinn_scores,neutral,subjective,t="Afinn Lexicon")
##plotHistogram(nrc1_scores,neutral,subjective,t="NRC v1 Lexicon")
##plotHistogram(nrc2_scores,neutral,subjective,t="NRC v2 Lexicon")
##plotHistogram(nrc3_scores,neutral,subjective,t="NRC v3 Lexicon")
##plotHistogram(nrc4_scores,neutral,subjective,t="NRC v4 Lexicon")
##plotHistogram(nrc5_scores,neutral,subjective,t="NRC v5 Lexicon")
##plotHistogram(mpqa_scores,neutral,subjective,t="MPQA Lexicon")
##plotHistogram(swn_scores,neutral,subjective,t="SentiWordNet Lexicon")

##plotDistribution(socal_scores,neutral,subjective,t="Socal Lexicon")
##plotDistribution(minqinghu_scores,neutral,subjective,t="Minqing Hu Lexicon")
##plotDistribution(afinn_scores,neutral,subjective,t="Afinn Lexicon")
##plotDistribution(nrc1_scores,neutral,subjective,t="NRC v1 Lexicon")
##plotDistribution(nrc2_scores,neutral,subjective,t="NRC v2 Lexicon")
##plotDistribution(nrc3_scores,neutral,subjective,t="NRC v3 Lexicon")
##plotDistribution(nrc4_scores,neutral,subjective,t="NRC v4 Lexicon")
##plotDistribution(nrc5_scores,neutral,subjective,t="NRC v5 Lexicon")
##plotDistribution(mpqa_scores,neutral,subjective,t="MPQA Lexicon")
##plotDistribution(swn_scores,neutral,subjective,t="SentiWordNet Lexicon")
