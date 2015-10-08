from tsvfiles import tsvreader
from nltk import FreqDist, Text, word_tokenize,bigrams
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import pylab
from tokenizers import twokenize
import numpy as np
from scipy.stats import norm

def printList(l):
    for word in l:
        print word

def subList(messages,labels,s):
    sub=[]
    for i in range(0,len(messages)):
        if labels[i]==s:
            sub.append(messages[i])

    return sub

def removeStopwords(t, n, label="No label"):
    frequency = FreqDist(t)
    s1 = frequency.most_common(n)
    s1 = [x[0] for x in s1]
##
##    print "Top "+str(n)+" stopwords of "+label
##    printList(stopwords)
##    
##    new_text = [x for x in t if x not in stopwords]
##    return new_text
    s2 = stopwords.words("english")
    s = set(s1+s2)
    return [x for x in t if x not in s]
##    return [x for x in t if x not in s2]

def plotLengthDistribution(l,t):
    data = np.array(l)

    #fit a normal distribution to the data
    mu, std = norm.fit(data)

    #plot histogram
    plt.hist(data, normed=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = t+"\nFit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)

    plt.show()
    
#def main():
    
#read labels and messages from dataset
dataset = "datasets/tweets#2015.tsv"
labels, messages = tsvreader.opentsv(dataset)
neutral_messages = subList(messages,labels,"neutral")
positive_messages = subList(messages,labels,"positive")
negative_messages = subList(messages,labels,"negative")

remove_stopwords = True

number_of_stopwords = 20
number_of_bigrams = 10
number_of_most_common = 10

#average length of messages
average_length_all = sum([len(x) for x in messages])/len(messages)
average_length_neutral = sum([len(x) for x in neutral_messages])/len(neutral_messages)
average_length_positive = sum([len(x) for x in positive_messages])/len(positive_messages)
average_length_negative = sum([len(x) for x in negative_messages])/len(negative_messages)

#get the text of messages

#tokenization with ntlk tokenizer
#text_all = Text(word_tokenize(str(messages)))
#text_neutral = Text(word_tokenize(str(neutral_messages)))
#text_positive = Text(word_tokenize(str(positive_messages)))
#text_negative = Text(word_tokenize(str(negative_messages)))

#tokenization with Noah's Ark Simple tokenizer
text_all = Text(twokenize.simpleTokenize(str(messages)))
text_neutral = Text(twokenize.simpleTokenize(str(neutral_messages)))
text_positive = Text(twokenize.simpleTokenize(str(positive_messages)))
text_negative = Text(twokenize.simpleTokenize(str(negative_messages)))

#remove stopwords
if remove_stopwords:
    text_all = removeStopwords(text_all, number_of_stopwords,"All")
    text_neutral = removeStopwords(text_neutral, number_of_stopwords,"Neutral")
    text_positive = removeStopwords(text_positive, number_of_stopwords,"Positive")
    text_negative = removeStopwords(text_negative, number_of_stopwords,"Negative")
    
    #print "***Top "+str(number_of_stopwords)+" stopwords removed***"
    print "Stopwords removed"

#total words of messages
#print "Total words of All messages : "+str(len(text_all))

#print "Total words of Neutral messages : "+str(len(text_neutral))

#print "Total words of Positive messages : "+str(len(text_positive))

#print "Total words of Negative messages : "+str(len(text_negative))

#frequency of every word in messages
freq_all = FreqDist(text_all)
freq_neutral = FreqDist(text_neutral)
freq_positive = FreqDist(text_positive)
freq_negative = FreqDist(text_negative)

#most common words
##print "Most common words in All messages"
##printList(freq_all.most_common(number_of_most_common))
##
##print "Most common words in Neutral messages"
##printList(freq_neutral.most_common(number_of_most_common))
##
##print "Most common words in Positive messages"
##printList(freq_positive.most_common(number_of_most_common))
##
##print "Most common words in Negative messages"
##printList(freq_negative.most_common(number_of_most_common))

#most common bigrams
##print "Most common bigrams in All messages"
##bigrams_all = list(bigrams(text_all))
##printList(Counter(bigrams_all).most_common(number_of_bigrams))
##print "Most common bigrams in Neutral messages"
##bigrams_neutral = list(bigrams(text_neutral))
##printList(Counter(bigrams_neutral).most_common(number_of_bigrams))
##print "Most common bigrams in Positive messages"
##bigrams_positive = list(bigrams(text_positive))
##printList(Counter(bigrams_positive).most_common(number_of_bigrams))
##print "Most common bigrams in Negative messages"
##bigrams_negative = list(bigrams(text_negative))
##printList(Counter(bigrams_negative).most_common(number_of_bigrams))

#total number of neutral,positive,negative messages
total = len(messages)
total_neutral = len([s for s in labels if s=="neutral"])
total_positive = len([s for s in labels if s=="positive"])
total_negative = len([s for s in labels if s=="negative"])

#plot number of messages
slices = [total_neutral,total_positive,total_negative]
fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot(111)
cmap = plt.cm.prism
pie_labels = ["neutral :"+str(total_neutral),"positive : "+str(total_positive),"negative : "+str(total_negative)]
ax.pie(slices,labels=pie_labels,labeldistance=1.05)
ax.set_title("Total number of messages : "+str(total))
plt.show()

#plot length of messages
pylab.xlabel("Category")
pylab.ylabel("Average length")
pylab.title("Average length of messages")
plt.bar(range(4),[average_length_all,average_length_neutral,average_length_positive,average_length_negative],align="center")
plt.xticks(range(4),["All ("+str(average_length_all)+")","Neutral ("+str(average_length_neutral)+")","Positive ("+str(average_length_positive)+")","Negative ("+str(average_length_negative)+")"])
plt.show()

#frequency plot
#freq_all.plot(20,cumulative=True)
#freq_neutral.plot(20,cumulative=True)
#freq_positive.plot(20,cumulative=True)
#freq_negative.plot(20,cumulative=True)

#commmon words for subjective and objective messages
text_objective = text_neutral
text_subjective = list(text_positive) + list(text_negative)
freq_objective = FreqDist(text_objective)
freq_subjective = FreqDist(text_subjective)

common_words_objective = freq_objective.most_common(number_of_most_common)
common_words_subjective = freq_subjective.most_common(number_of_most_common)

objective_labels = [x[0] for x in common_words_objective]
subjective_labels = [x[0] for x in common_words_subjective]

labels = list(set(objective_labels+subjective_labels))

objective_keys = [freq_objective.get(x) for x in labels]
subjective_keys = [freq_subjective.get(x) for x in labels]

for i in range(0,len(labels)):
    if objective_keys[i]==None:
        objective_keys[i]=0

    if subjective_keys[i]==None:
        subjective_keys[i]=0

X=np.arange(len(labels))
#plt.bar(X+0.00,objective_keys,color="c",width=0.35,label="Objective messages")
#plt.bar(X+0.25,subjective_keys,color="r",width=0.35,label="Subjective messages")
plt.bar(X+0.00,[x/float(total_neutral)*100 for x in objective_keys],color="c",width=0.35,label="Objective messages")
plt.bar(X+0.25,[x/float(total_negative+total_positive)*100 for x in subjective_keys],color="r",width=0.35,label="Subjective messages")
#plt.xticks([x+0.25 for x in range(len(labels))],labels)
plt.xticks([x+0.25 for x in range(len(labels))],([x for x in labels]),size=15)
plt.ylabel("%")
plt.legend(loc="upper right")
plt.title("Most common words(stopwords removed)")

plt.show()

#distribution of length of messages
len_all = [len(x) for x in messages]
len_neutral = [len(x) for x in neutral_messages]
len_positive = [len(x) for x in positive_messages]
len_negative = [len(x) for x in negative_messages]
len_subjective = len_positive + len_negative

plotLengthDistribution(len_all,"All messages")
plotLengthDistribution(len_neutral,"Objective messages")
plotLengthDistribution(len_positive,"Positive messages")
plotLengthDistribution(len_negative,"Negative messages")
plotLengthDistribution(len_subjective,"Subjective messages")

    
#if __name__ == "__main__":
#    main()
