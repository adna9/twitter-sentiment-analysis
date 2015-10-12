from nltk.corpus import sentiwordnet as s

#class representing SentiWordNet Lexicon
class SentiWordNetLexicon():

    #availble pos tags in SentiWordNet
    senti_tags = ["n", "a", "v", "r"]

    #empty constructor
    def __init__(self):
        pass

    #compute score of message
    def score(self,tokens,pos_tags):
        total = 0.0

        for i in range(0,len(pos_tags)) :
            #find word sentiment score
            total += self.findSentiment(tokens[i],pos_tags[i])

        return total

    #find word's sentiment score
    def findSentiment(self,token,pos_tag):
            #check if the token is in lexicon
            #n a v r
            #if s.senti_synsets(token) > 0 :
            synsets = s.senti_synsets(token)
            
            #if (pos_tag.lower() in SentiWordNetLexicon.senti_tags) :
            if len(synsets)>0 :
                #find available synsets for the specific pos

                #synsets = s.senti_synsets(token,pos_tag.lower())
                #if len(synsets)==0: return 0.0

                #calculate score for all sentimens(neutral,positive,negative)
                neutral = 0
                positive = 0
                negative = 0

                #average score of all synsets (allagi an vrethei kaliteros tropos)
                for synset in synsets:
                    neutral += synset.obj_score()
                    positive += synset.pos_score()
                    negative += synset.neg_score()

                neutral = neutral/float(len(synsets))
                positive = positive/float(len(synsets))
                negative = negative/float(len(synsets))

                #return sentiment with max score
                if max(neutral,positive,negative) == neutral :
                    return 0.0
                elif max(neutral,positive,negative) == positive :
                    return -1.0
                else:
                    return +1.0                
            else:
                return 0.0


    #compute the number of tokens(words) that appear in the lexicon
    def getNumberOfAppearances(self,tokens):
        total = 0
        for token in tokens:
            if len(s.senti_synsets(token)) > 0:
                total += 1

        return total
