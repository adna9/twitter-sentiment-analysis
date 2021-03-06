#check if a message has negation
def hasNegation(tokens,negationList):
    for token in tokens:
        if token in negationList:
            return 1

    return 0

#check if message has negation preceding words from lexicon
def hasNegationPrecedingLexicon(lexicon,tokens,negationList):
    for i in range(0,len(tokens)):
        if tokens[i] in lexicon.lexicon:
            #get 5 preciding tokens
            subList = tokens[max(0,i-5):i]

            #check if these tokens are in the lexicon
            for token in subList:
                if token in negationList:
                    return 1

    return 0
            
    
#calculate the existence of happy emoticons
def happy_emoticons(tokens):
    
    emoticons = [':)',':-)',':o)',':]',':3',':c)',':>','=]','8)','=)',':}',':^)']
    for token in tokens :
        if token in emoticons :
            return 1 
        
    return 0


#calculate the existence of sad emoticons
def sad_emoticons(tokens):
    
    emoticons = [':(',':-(','>:[',':-c',':c',':-<',':<',':-[',':[',':{']
    
    for token in tokens :
        if token in emoticons :
            return 1 
        
    return 0


def numberOfPronounVerbVerb(trigrams):
  return len([a for a,b,c in trigrams if a=="O" and b=="V" and c =="V"])

def numberOfVerbDeterminerNoun(trigrams):
  return len([a for a,b,c in trigrams if a=="V" and b=="D" and c =="N"])

def numberOfPositionDeterminerNoun(trigrams):
  return len([a for a,b,c in trigrams if a=="P" and b=="D" and c =="N"])

def numberOfAdjectiveNoun(bigrams):

    return len([b for b,t in bigrams if b=="A" and t=="N"])
    
def numberOfVerbVerb(bigrams):

    return len([b for b,t in bigrams if b=="V" and t=="V"])


def numberOfNounVerb(bigrams):
    
   return len([b for b,t in bigrams if b=="N" and t=="V"])

def numberOfVerbAdverb(bigrams):
    
    return len([b for b,t in bigrams if b=="V" and t=="R"])


def numberOfAdverbVerb(bigrams):
    return len([b for b,t in bigrams if b=="R" and t=="V"])


def numberOfPronounVerb(bigrams):
  return len([b for b,t in bigrams if b=="O" and t=="V"])
