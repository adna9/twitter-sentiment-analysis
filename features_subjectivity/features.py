import numpy as np
from morphologicalFeatures import *
from posBasedFeatures import *

#return feautures of a list of messages as an array
def getFeatures(messages,tokens,pos,slangDictionary):
    #initialize empty list with features for all message
    features = []

    #calculate features for every message
    for i in range(0,len(messages)):
        
        #list with features for one message
        f = calculateFeatures(messages[i],tokens[i],pos[i],slangDictionary)

        #add f to features
        features.append(f)

    #convert features list to numpy array
    features_array = np.array(features)
    #return test array , with no actual features
    #features_array = np.random.rand(len(messages),10)

    #return result
    return features_array

#calculate features for a message
def calculateFeatures(message,tokens,pos,slangDictionary):
    f=[]

    #Morphological Features
    
    #feature 1 - existance of enlogated tokens in message e.g. "baaad"
    x = hasElongatedWords(message)
    f.append(x)

    #feature 2 - the number of elongated tokens in the message
    x = numberOfElongatedWords(message)
    f.append(x)

    #feature 3 - existance of date expressions in message
    x = hasDateExpressions(message)
    f.append(x)

    #feature 4 - existance of time expressions in message
    x = hasTimeExpressions(message)
    f.append(x)

    #feature 5 - the number of tokens of the message that are fully capitalized
    x = countFullyCapitalizeTokens(tokens)
    f.append(x)

    #feature 6 - the number of tokens that are partially capitalized
    x = countPartiallyCapitalizeTokens(tokens)
    f.append(x)

    #feature 7 - the number of tokens that start with an upper case letter
    x = countUpper(tokens)
    f.append(x)

    #feature 8 - the number of exclamation marks in the message
    ex = countExclamationMarks(message)
    f.append(ex)

    #feature 9 - the number of question marks
    qu = countQuestionMarks(message)
    f.append(qu)

    #feature 10 - the sum of exclamation and question marks
    x = ex + qu
    f.append(x)

    #feauture 11 - the number of tokens containing only ellipsis (...)
    x = countEllipsis(tokens)
    f.append(x)

    #feature 12 - the existence of a subjective emoticon at the message's end
    x = hasEmoticonAtEnd(tokens[len(tokens)-1],pos[len(pos)-1])
    f.append(x)

    #feature 13 - the existence of an ellipsis and a link (URL) at the message's end
    x = hasUrlOrEllipsisAtEnd(tokens[len(tokens)-1],pos[len(pos)-1])
    f.append(x)

    #feature 14 - the existence of an exclamation mark at the message's end
    x = hasExclamationMarkAtEnd(tokens[len(tokens)-1])
    f.append(x)

    #feature 15 - the existence of a question mark at the message's end
    x = hasQuestionMarkAtEnd(tokens[len(tokens)-1])
    f.append(x)

    #feature 16 - the existence of a question or an exclamation mark at the message's end
    x = hasQuestionOrExclamationMarkAtEnd(tokens[len(tokens)-1])
    f.append(x)

    #feature 17 - the existence of slang
    x = hasSlang(tokens,slangDictionary)
    f.append(x)

    #Pos Based Features

    #feature 18 - the number of adjectives in the message
    x = numberOfAdjectives(pos)
    f.append(x)

    #feature 19 - the number of adverbs
    x = numberOfAdverbs(pos)
    f.append(x)

    #feature 20 - the number of interjections
    x = numberOfIntejections(pos)
    f.append(x)

    #feature 21 - the number of verbs
    x = numberOfVerbs(pos)
    f.append(x)

    #feature 22 - the number of nouns
    x = numberOfNouns(pos)
    f.append(x)

    #feature 23 - the number of proper nouns
    x = numberOfProperNouns(pos,tokens)
    f.append(x)

    #feature 24 - the number of urls
    x = numberOfUrls(pos,tokens)
    f.append(x)

    #feature 25 - the number of subjective emoticons
    x = numberOfSubjectiveEmoticons(pos,tokens)
    f.append(x)

    




    return f
    
