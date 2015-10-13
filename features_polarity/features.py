import numpy as np
from morphologicalFeatures import *
from posBasedFeatures import *
from bigramsFeatures import *
from trigramsFeatures import *


#return feautures of a list of messages as an array
def getFeatures(messages,tokens,pos,bigrams,trigrams,slangDictionary,lex):
    #initialize empty list with features for all message
    features = []

    #calculate features for every message
    for i in range(0,len(messages)):
        
        #list with features for one message
        f = calculateFeatures(messages[i],tokens[i],pos[i],bigrams[i],trigrams[i],slangDictionary,lex)

        #add f to features
        features.append(f)

    #convert features list to numpy array
    features_array = np.array(features)
    #return test array , with no actual features
    #features_array = np.random.rand(len(messages),10)

    #return result
    return features_array

#calculate features for a message
def calculateFeatures(message,tokens,pos,bigrams,trigrams,slangDictionary,lex):
    f=[]

    #bigrams
    x=numberOfAdjectiveNoun(bigrams)
    f.append(x)
    x=numberOfVerbVerb(bigrams)
    f.append(x)
    x=numberOfNounVerb(bigrams)
    f.append(x)
    x=numberOfAdverbVerb(bigrams)
    f.append(x)
    x=numberOfVerbAdverb(bigrams)
    f.append(x)
    x=numberOfPronounVerb(bigrams)
    f.append(x)

    #trigrams
    x=numberOfPronounVerbVerb(trigrams)
    f.append(x)
    x=numberOfVerbDeterminerNoun(trigrams)
    f.append(x)
    x=numberOfPositionDeterminerNoun(trigrams)
    f.append(x)
    
    
    #posBasedFeatures
    x = numberOfAdjectives(pos)
    f.append(x)
    x = numberOfAdverbs(pos)
    f.append(x)
    x = numberOfIntejections(pos)
    f.append(x)
    x = numberOfVerbs(pos)
    f.append(x)
    x = numberOfNouns(pos)
    f.append(x)
    x = numberOfProperNouns(pos,tokens)
    f.append(x)

    #x = presense of happy emoticon
    x=happy_emoticons(tokens)
    f.append(x)
    #x = presence of sad_emoticon
    x=sad_emoticons(tokens)
    f.append(x)

    #lexiconscore
    x=senti_score(tokens, lex)
    f.append(x)

    #morphologicalFeatures
    x=numOfSlangs(tokens,slangDictionary) 
    f.append(x)
    x=countEllipsis(tokens)
    f.append(x)
    x=countExclamationMarks(message)
    f.append(x)
    x=numberOfElongatedWords(message)
    f.append(x)
    x=countPartiallyCapitalizeTokens(tokens)
    f.append(x)

    return f
    
