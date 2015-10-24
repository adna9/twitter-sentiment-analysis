#class representing the MPQA Lexicon
class MPQALexicon():

    #lexicon directory
    directory = "lexicons/MPQA/"

    #lexicon files
    file1 = "subjclueslen1-HLTEMNLP05.tff"

    #constructor
    def __init__(self,option):
        #initialize Lexicon
        self.lexicon = []

        self.loadLexicon(option)

    #load a lexicon file
    def loadLexicon(self,option):
        f = open(MPQALexicon.directory+MPQALexicon.file1,"r")
        
        if option == 0:
            self.loadLexicon0(f)
        elif option == 1:
            self.loadLexicon1(f)
        elif option == 2:
            self.loadLexicon2(f)
        elif option == 3:
            self.loadLexicon3(f)
        elif option == 4:
            self.loadLexicon4(f)
        elif option == 5:
            self.loadLexicon5(f)
        elif option == 6:
            self.loadLexicon6(f)
        elif option == 7:
            self.loadLexicon7(f)
        else:
            print "Lexicon unavailable"

        f.close()

    #load Lexicon that contains strong subjective words with positive prior polarity
    def loadLexicon0(self,f):
        for line in f.readlines():
            s = line.split(" ")

            if (s[0].split("=")[1] == "strongsubj" and s[5].split("=")[1] == "positive\n"):
                self.lexicon.append(s[2].split("=")[1])
                
        
    #load Lexicon that contains strong subjective words with negative prior polarity
    def loadLexicon1(self,f):
        for line in f.readlines():
            s = line.split(" ")

            if (s[0].split("=")[1] == "strongsubj" and s[5].split("=")[1] == "negative\n"):
                self.lexicon.append(s[2].split("=")[1])

    #load Lexicon that contains strong subjective words with either positive or negative prior polarity
    def loadLexicon2(self,f):
        for line in f.readlines():
            s = line.split(" ")
            
            if (s[0].split("=")[1] == "strongsubj" and (s[5].split("=")[1] == "negative\n" or s[5].split("=")[1] == "positive\n")):
                self.lexicon.append(s[2].split("=")[1])

    #load Lexicon that contains strong subjective words with neutral prior polarity
    def loadLexicon3(self,f):
        for line in f.readlines():
            s = line.split(" ")

            if (s[0].split("=")[1] == "strongsubj" and s[5].split("=")[1] == "neutral\n"):
                self.lexicon.append(s[2].split("=")[1])

    #load Lexicon that contains weak subjective words with positive prior polarity
    def loadLexicon4(self,f):
        for line in f.readlines():
            s = line.split(" ")
            
            if (s[0].split("=")[1] == "weaksubj" and s[5].split("=")[1] == "positive\n"):
                self.lexicon.append(s[2].split("=")[1])

    #load Lexicon that contains weak subjective words with negative prior polarity
    def loadLexicon5(self,f):
        for line in f.readlines():
            s = line.split(" ")

            if (s[0].split("=")[1] == "weaksubj" and s[5].split("=")[1] == "negative\n"):
                self.lexicon.append(s[2].split("=")[1])

    #load Lexicon that contains weak subjective words with either positive or negative prior polarity
    def loadLexicon6(self,f):
        for line in f.readlines():
            s = line.split(" ")

            if (s[0].split("=")[1] == "weaksubj" and (s[5].split("=")[1] == "negative\n" or s[5].split("=")[1] == "positive\n")):
                self.lexicon.append(s[2].split("=")[1])

    #load Lexicon that contains weak subjective words with neutral prior polarity
    def loadLexicon7(self,f):
        for line in f.readlines():
            s = line.split(" ")

            if (s[0].split("=")[1] == "weaksubj" and s[5].split("=")[1] == "neutral\n"):
                self.lexicon.append(s[2].split("=")[1])

    #compute the number of tokens(words) that appear in the lexicon
    def getNumberOfAppearances(self,tokens):
        total = 0
        for token in tokens:
            if token in self.lexicon:
                total+=1

        return total
