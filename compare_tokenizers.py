from tsvfiles import tsvreader
from nltk import word_tokenize
from tokenizers import twokenize
import matplotlib.pyplot as plt

#read labels and messages from dataset
dataset = "datasets/tweets#2015.tsv"
labels, messages = tsvreader.opentsv(dataset)

total = 0
noahs_total = 0

for message in messages:
    nlkt_tok = word_tokenize(message)
    noahs_simple = twokenize.simpleTokenize(message)
    noahs_nohmlt = twokenize.tokenize(message)
    noahs_raw = twokenize.tokenizeRawTweetText(message)

    #check when ntlk and noah's ark tokenizer "agree"
    if(nlkt_tok==noahs_simple or nlkt_tok==noahs_nohmlt or nlkt_tok==noahs_raw):
        total+=1

    #check when the 3 noah's ark tokenizers "agree"
    if(noahs_simple==noahs_nohmlt and noahs_simple==noahs_raw and noahs_nohmlt==noahs_raw):
        noahs_total+=1
            
#plot pie
slices = [total,len(messages)-total]
fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot(111)
cmap = plt.cm.prism
pie_labels = ["messages with same tokens :"+str(total),"messages with different tokens : "+str(len(messages)-total)]
ax.pie(slices,labels=pie_labels,labeldistance=1.05)
ax.set_title("NLTK and Noah's Ark Tokenizers \n\nTotal number of messages : "+str(len(messages)))
plt.show()

#plot pie
slices = [noahs_total,len(messages)-noahs_total]
fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot(111)
cmap = plt.cm.prism
pie_labels = ["messages with \nsame tokens :\n"+str(noahs_total),"messages with \ndifferent tokens : \n"+str(len(messages)-noahs_total)]
ax.pie(slices,labels=pie_labels,labeldistance=1.05)
ax.set_title("Noah's Ark Tokenizers \n\nTotal number of messages : "+str(len(messages)))
plt.show()
