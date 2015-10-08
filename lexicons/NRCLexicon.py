from collections import Counter
from nltk import bigrams

#class representing the NRC Lexicon
class NRCLexicon():

    #lexicon directory
    directory="lexicons/NRC/"

    #constructor
    def __init__(self,lexicon):
        #initialize dictionaries - 2 dictionaries(unigrams and bigrams)
        self.d_unigrams = {}
        self.d_bigrams = {}

        #select which lexicon to load
        if lexicon == 0 :
            self.loadHashtagLexicon1()
        elif lexicon == 1:
            self.loadHashtagLexicon2()
        elif lexicon == 2 :
            self.loadMaxDiffTwitterLexicon()
        elif lexicon == 3 :
            self.loadSentiment140Lexicon1()
        elif lexicon == 4 :
            self.loadSentiment140Lexicon2()
        else:
            print "Lexicon unavailable, please load another one"

    #HashtagSentimentAffLexNegLex
    def loadHashtagLexicon1(self):
        folder = "HashtagSentimentAffLexNegLex/"
        file1 = "HS-AFFLEX-NEGLEX-unigrams.txt"
        file2 = "HS-AFFLEX-NEGLEX-bigrams.txt"

        #clear previous dictionaries
        self.clearDictionaries()

        #load unigrams
        self.loadUnigrams(NRCLexicon.directory+folder+file1)

        #load bigrams
        self.loadBigrams(NRCLexicon.directory+folder+file2)

    #NRC-Hashtag-Sentiment-Lexicon-v0.1
    def loadHashtagLexicon2(self):
        folder = "NRC-Hashtag-Sentiment-Lexicon-v0.1/"
        file1 = "unigrams-pmilexicon.txt"
        file2 = "bigrams-pmilexicon.txt"

        #clear previous dictionaries
        self.clearDictionaries()

        #load unigrams
        self.loadUnigrams(NRCLexicon.directory+folder+file1)

        #load bigrams
        self.loadBigrams(NRCLexicon.directory+folder+file2)

    #MaxDiff-Twitter-Lexicon
    def loadMaxDiffTwitterLexicon(self):
        folder = "MaxDiff-Twitter-Lexicon/"
        file1 = "Maxdiff-Twitter-Lexicon_-1to1.txt"

        #clear previous dictionaries
        self.clearDictionaries()

        #load unigrams - reverse = true due to the .txt file format
        self.loadUnigrams(NRCLexicon.directory+folder+file1,True)

        #this lexicon has no bigrams so d_bigrams remains empty

    #Sentiment140AffLexNegLex
    def loadSentiment140Lexicon1(self):
        folder = "Sentiment140AffLexNegLex/"
        file1 = "S140-AFFLEX-NEGLEX-unigrams.txt"
        file2 = "S140-AFFLEX-NEGLEX-bigrams.txt"

        #clear previous dictionaries
        self.clearDictionaries()

        #load unigrams
        self.loadUnigrams(NRCLexicon.directory+folder+file1)

        #load bigrams
        self.loadBigrams(NRCLexicon.directory+folder+file2)
        
    #Sentiment140-Lexicon-v0.1
    def loadSentiment140Lexicon2(self):
        folder = "Sentiment140-Lexicon-v0.1/"
        file1 = "unigrams-pmilexicon.txt"
        file2 = "bigrams-pmilexicon.txt"

        #clear previous dictionaries
        self.clearDictionaries()

        #load unigrams
        self.loadUnigrams(NRCLexicon.directory+folder+file1)

        #load bigrams
        self.loadBigrams(NRCLexicon.directory+folder+file2)

    #clear dictionaries
    def clearDictionaries(self):
        if self.d_unigrams is not None:
            self.d_unigrams.clear()

        if self.d_bigrams is not None:
            self.d_bigrams.clear()
            
    #load unigrams lexicon
    def loadUnigrams(self,path,reverse=False):
        f = open(path)

        for line in f.readlines():
            key = line.split("\t")[0]
            value = line.split("\t")[1]

            if reverse:
                self.d_unigrams[value]=float(key)
            else:
                self.d_unigrams[key]=float(value)

        f.close()
                
    #load bigrams lexicon
    def loadBigrams(self,path):
        f = open(path)

        for line in f.readlines():
            key = line.split("\t")[0]
            value = line.split("\t")[1]

            #represent bigrams as tuples
            t1 = key.split(" ")[0]
            t2 = key.split(" ")[1]
            tpl = (t1,t2)
            self.d_bigrams[tpl]=float(value)

        f.close()

    #compute score of a message
    def score(self,tokens):
        total = 0.0
        #score for unigrams
        for token in tokens:
            total += self.d_unigrams.get(token,0.0)

        #score for bigrams, if bigrams exist
        if len(self.d_bigrams)>0 :
            #list with bigrams of the message
            bigrams_list = Counter(list(bigrams(tokens))).keys()
            for bigram in bigrams_list :
                total += self.d_bigrams.get(bigram,0.0)
            
        return total
