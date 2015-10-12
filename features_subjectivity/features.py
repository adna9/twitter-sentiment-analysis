import numpy as np
from morphologicalFeatures import *
from posBasedFeatures import *
from lexiconBasedFeatures import *
from posBigramsFeatures import *

#return feautures of a list of messages as an array
def getFeatures(messages,tokens,pos,slangDictionary,lexicons,pos_bigrams,pos_bigrams_scores_objective,pos_bigrams_scores_subjective):
    #initialize empty list with features for all message
    features = []

    #calculate features for every message
    for i in range(0,len(messages)):
        
        #list with features for one message
        f = calculateFeatures(messages[i],tokens[i],pos[i],slangDictionary,lexicons,pos_bigrams[i],pos_bigrams_scores_objective,pos_bigrams_scores_subjective)

        #add f to features
        features.append(f)

    #convert features list to numpy array
    features_array = np.array(features)
    #return test array , with no actual features
    #features_array = np.random.rand(len(messages),10)

    #return result
    return features_array

#calculate features for a message
def calculateFeatures(message,tokens,pos,slangDictionary,lexicons,pos_bigrams,pos_bigrams_scores_objective,pos_bigrams_scores_subjective):
    f=[]
    #Morphological Features
    
    #existance of enlogated tokens in message e.g. "baaad"
    x = hasElongatedWords(message)
    f.append(x)

    #the number of elongated tokens in the message
    x = numberOfElongatedWords(message)
    f.append(x)

    #existance of date expressions in message
    x = hasDateExpressions(message)
    f.append(x)

    #existance of time expressions in message
    x = hasTimeExpressions(message)
    f.append(x)

    #the number of tokens of the message that are fully capitalized
    x = countFullyCapitalizeTokens(tokens)
    f.append(x)

    #the number of tokens that are partially capitalized
    x = countPartiallyCapitalizeTokens(tokens)
    f.append(x)

    #the number of tokens that start with an upper case letter
    x = countUpper(tokens)
    f.append(x)

    #the number of exclamation marks in the message
    ex = countExclamationMarks(message)
    f.append(ex)

    #the number of question marks
    qu = countQuestionMarks(message)
    f.append(qu)

    #the sum of exclamation and question marks
    x = ex + qu
    f.append(x)

    #the number of tokens containing only ellipsis (...)
    x = countEllipsis(tokens)
    f.append(x)

    #the existence of a subjective emoticon at the message's end
    x = hasEmoticonAtEnd(tokens[len(tokens)-1],pos[len(pos)-1])
    f.append(x)

    #the existence of an ellipsis and a link (URL) at the message's end
    x = hasUrlOrEllipsisAtEnd(tokens[len(tokens)-1],pos[len(pos)-1])
    f.append(x)

    #the existence of an exclamation mark at the message's end
    x = hasExclamationMarkAtEnd(tokens[len(tokens)-1])
    f.append(x)

    #the existence of a question mark at the message's end
    x = hasQuestionMarkAtEnd(tokens[len(tokens)-1])
    f.append(x)

    #the existence of a question or an exclamation mark at the message's end
    x = hasQuestionOrExclamationMarkAtEnd(tokens[len(tokens)-1])
    f.append(x)

    #the existence of slang
    x = hasSlang(tokens,slangDictionary)
    f.append(x)

    #Pos Based Features

    #the number of adjectives in the message
    x = numberOfAdjectives(pos)
    f.append(x)

    #the number of adverbs
    x = numberOfAdverbs(pos)
    f.append(x)

    #the number of interjections
    x = numberOfIntejections(pos)
    f.append(x)

    #the number of verbs
    x = numberOfVerbs(pos)
    f.append(x)

    #the number of nouns
    x = numberOfNouns(pos)
    f.append(x)

    #the number of proper nouns
    x = numberOfProperNouns(pos,tokens)
    f.append(x)

    #the number of urls
    x = numberOfUrls(pos,tokens)
    f.append(x)

    #the number of subjective emoticons
    x = numberOfSubjectiveEmoticons(pos,tokens)
    f.append(x)

    #Pos Bigrams Features
    
    #the average,maximun,minium f1 score for the messages pos bigrams for objective messages
    average, maximum, minimum = F1PosBigramsScore(pos_bigrams,pos_bigrams_scores_objective)
    f.append(average)
    f.append(maximum)
    f.append(minimum)

    #the average,maximun,minium f1 score for the messages pos bigrams for objective messages
    average, maximum, minimum = F1PosBigramsScore(pos_bigrams,pos_bigrams_scores_subjective)
    f.append(average)
    f.append(maximum)
    f.append(minimum)
    

    # Lexicon Based Features

    #iterate for every lexicon
    for lexicon in lexicons:
        #score of lexicon (total score of all words)
        x = sumOfScores(lexicon,message,tokens,pos)
        f.append(x)

        #average of scores
        f.append(x/float(len(tokens)))

        #max score of words
        x = maxOfScores(lexicon,tokens,pos)
        f.append(x)

        #min score of words
        x = minOfScores(lexicon,tokens,pos)
        f.append(x)

        #the count of words of the message that appear in the lexicon
        x = numberOfAppearances(lexicon,tokens)
        f.append(x)

        #the score of the last word of the message
        x = scoreOfLastWord(lexicon,tokens[len(tokens)-1],pos[len(pos)-1])
        f.append(x)

        #the score of the last word of the message that appears in the lexicon
        x = scoreOfLastWordAppearedInLexicon(lexicon,tokens,pos)
        f.append(x)

        




    return f
    
