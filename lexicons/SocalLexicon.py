#class representing the Socal Lexicon
class SocalLexicon():
    
    #lexicon directory
    directory = "lexicons/SO-CAL/"
    #directory = "SO-CAL/"

    #lexicon files
    file1 = "adj_dictionary1.11.txt"
    file2 = "adv_dictionary1.11.txt"
    file3 = "int_dictionary1.11.txt"
    file4 = "noun_dictionary1.11.txt"
    file5 = "verb_dictionary1.11.txt"
    
    #constructor
    def __init__(self):
        #initialize dictionary
        self.d = {}
        #load Lexicon
        self.loadLexicon()

    #load a lexicon file
    def loadLexiconFile(self,filename):
        f = open(SocalLexicon.directory+filename,"r")

        for line in f.readlines():
            key = line.split("\t")[0]
            value = line.split("\t")[1]
            self.d[key]=float(value)

        f.close()

    #load SO-CAL Lexicon        
    def loadLexicon(self):
        #initialize dictionary
        #d = {}

        #load lexicons
        self.loadLexiconFile(SocalLexicon.file1)
        self.loadLexiconFile(SocalLexicon.file2)
        self.loadLexiconFile(SocalLexicon.file3)
        self.loadLexiconFile(SocalLexicon.file4)
        self.loadLexiconFile(SocalLexicon.file5)

        #return dictionary
        #return d

    #compute score of a message
    def score(self,tokens):
        total = 0.0
        for token in tokens:
            total += self.d.get(token,0.0)
            
        return total

    #compute the number of tokens(words) that appear in the lexicon
    def getNumberOfAppearances(self,tokens):
        total = 0
        for token in tokens:
            if self.d.has_key(token):
                total+=1

        return total
