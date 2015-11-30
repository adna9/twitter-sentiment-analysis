import numpy as np
import sys
sys.path.append('../features')
#from features import morphologicalFeatures,posBasedFeatures
from features.morphologicalFeatures import *
from features.posBasedFeatures import *

class Glove() :

    directory = "embeddings/Glove/"
    
    file1= "glove.twitter.27B.25d.txt"
    #file1 = "glove.twitter.27B.100d.txt"
    #file1 = "glove.twitter.27B.50d.txt"
   
    
    #constructor
    def __init__(self):
        #initialize dictionary

        self.embeddings={}
        self.size=25
        #self.size=100
        #self.size=50
        
        #load embeddings
        self.loadEmbeddings()

    #load Slang Dictionary
    def loadEmbeddings(self):

        f = open(Glove.directory+Glove.file1,"r")
       
        for line in f.readlines():
            
            line = line.decode('utf8')

            #remove \n characters
            line = line.rstrip()
            
            word = line.split(" ")[0]
            
            emb = [float(x) for x in line.split(" ")[1:]]
          
            self.embeddings[word] = emb

        f.close()

    def findWordEmbeddings(self,word):
        #return self.embeddings.get(word,np.zeros(self.size))
        return self.embeddings.get(word,0)


    def findCentroid(self,message,tokens,pos_tags):
        counter=0
        
        #initialize centroid
        centroid = [0 for x in range(0,self.size)]
        
        for token in tokens:
            #find embeddings for token
            emb = self.findWordEmbeddings(token)

            if emb!=0:   
                #add to centroid
                centroid = [centroid[i]+emb[i] for i in range(0,self.size)]
                counter+=1
            elif token[0]=="#":
                emb = self.embeddings.get("<hashtag>")
                centroid = [centroid[i]+emb[i] for i in range(0,self.size)]
                counter+=1

        for pos in pos_tags:
            emb=0
            
            if pos=="@":
                emb = self.embeddings.get("<user>")
            elif pos=="$":
                emb = self.embeddings.get("<number>")
            elif pos=="U":
                emb = self.embeddings.get("<url>")

            if emb!=0:
                centroid = [centroid[i]+emb[i] for i in range(0,self.size)]
                counter+=1

        if hasTimeExpressions(message)==1:
            emb = self.embeddings.get("<time>")
            centroid = [centroid[i]+emb[i] for i in range(0,self.size)]
            counter+=1


        positive_emoticons = numberOfPositiveEmoticons(tokens)
        for i in range(0,positive_emoticons):
            emb = self.embeddings.get("<smile>")
            centroid = [centroid[i]+emb[i] for i in range(0,self.size)]
            counter+=1

        negative_emoticons = numberOfNegativeEmoticons(tokens)
        for i in range(0,negative_emoticons):
            emb = self.embeddings.get("<sadface>")
            centroid = [centroid[i]+emb[i] for i in range(0,self.size)]
            counter+=1

        neutral_emoticons = numberOfNeutralEmoticons(tokens)
        for i in range(0,neutral_emoticons):
            emb = self.embeddings.get("<neutralface>")
            centroid = [centroid[i]+emb[i] for i in range(0,self.size)]
            counter+=1

        capitalized = countFullyCapitalizeTokens(tokens)
        for i in range(0,capitalized):
            emb = self.embeddings.get("<allcaps>")
            centroid = [centroid[i]+emb[i] for i in range(0,self.size)]
            counter+=1
            
        #divide with size
        centroid = [x/float(counter) for x in centroid]

        return centroid
