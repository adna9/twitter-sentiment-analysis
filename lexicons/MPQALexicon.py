#class representing the MPQA Lexicon
class MPQALexicon():

    #lexicon directory
    directory = "lexicons/MPQA/"

    #lexicon files
    file1 = "subjclueslen1-HLTEMNLP05.tff"

    #constructor
    def __init__(self):
        #initialize dictionaries

        #word - sentiment dictionary
        self.d = {}
        #word - stemmed dictionary (check if the word is stemmed)
        #self.s = {}

        self.loadLexicon()

    #load a lexicon file
    def loadLexicon(self):
        f = open(MPQALexicon.directory+MPQALexicon.file1,"r")

        for line in f.readlines():
            key = (line.split(" ")[2]).split("=")[1]
            value = str((line.split(" ")[5]).split("=")[1])
            value = value[0:len(value)-1]
            stemmed = (line.split(" ")[4]).split("=")[1]

            if value=="negative":
                value = -1
            elif value=="positive":
                value = +1
            else:
                value = 0

            if stemmed=="y":
                stemmed=1
            else:
                stemmed=0

            self.d[key]=float(value)
            #self.s[key]=stemmed
            
        f.close()

    #compute score of a message
    def score(self,tokens):
        total = 0

        for token in tokens:
            #check if word is stemmed
            #if self.s[token]==1:
                #if word is stemmed , we check all the words starting with our word
                #i.e if word = abuse we search for abuses,abused,abusing
            #elif:
            total += self.d.get(token,0.0)

        return total

    #compute the number of tokens(words) that appear in the lexicon
    def getNumberOfAppearances(self,tokens):
        total = 0
        for token in tokens:
            if self.d.has_key(token):
                total+=1

        return total
