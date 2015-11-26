import numpy as np 

class Glove() :

    directory = "embeddings/Glove/"
    
    file1= "glove.twitter.27B.25d.txt"
    #file1 = "test.txt"

    #constructor
    def __init__(self):
        #initialize dictionary

        self.embeddings={}
        self.size=25
        
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
            emb = [float(x) for x in line.split(" ")[1:(self.size+1)]]
          
            self.embeddings[word] = emb

        f.close()

    def findWordEmbeddings(self,word):
##        if word in self.embeddings.keys():
##            return self.embeddings[word]
##        
##        return np.zeros(self.size)
        return self.embeddings.get(word,np.zeros(self.size))

    def findCentroid(self,message):

        #initialize centroid
        centroid = [0 for x in range(0,self.size)]
        
        for token in message:
            #find embeddings for token
            emb = self.findWordEmbeddings(token)
            #add to centroid
            centroid = [centroid[i]+emb[i] for i in range(0,self.size)]

        #divide with size
        centroid = [x/float(len(message)) for x in centroid]

        return centroid
