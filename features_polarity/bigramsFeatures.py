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
