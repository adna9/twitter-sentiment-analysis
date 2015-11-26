import numpy as np 

class Glove() :

    directory = "lexicons/Glove/"

    file1= "glove.twitter.27B.25d.txt"

    #constructor
    def __init__(self):
        #initialize dictionary

        self.embeddings={}
        self.size=25
        
        #load Slang Dictionary
        self.loadDictionary()

    #load Slang Dictionary
    def loadDictionary(self):

        f = open(Glove.directory+Glove.file1,"r")
       
        for line in f.readlines():
            
            line = line.decode('utf8')

            word = line.split(" ")[0]
          
            self.embeddings[word]=[emb for emb in line.split(" ")[1:]]

        f.close()

    def findWordEmbeddings(self,word):

            
        if word in self.embeddings.keys():
            return self.embeddings[word]

        
        return np.zeros(self.size)
