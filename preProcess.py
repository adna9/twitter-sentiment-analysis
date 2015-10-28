from postaggers import arktagger
from nltk import bigrams
from utilities import *
import re
import enchant
from tokenizers import twokenize


def processMessage(message,slangDictionary,dictionary):

    #tokenize message
    tokens = twokenize.simpleTokenize(message)

    #the processed message
    process=""

    #pattern to squeeze elongated tokens
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)

    #Character replacement mapping
    d={}
    d["O"]="o"
    d["3"]="e"
    d["@"]="a"
    d["#"]="h"
    d[""]="ate"
    d["4"]="for"
    d["!"]="i"
    d["$"]="s"
    d["1"]="i"
    d["2"]="i"
    d["5"]="to"
    d["7"]="s"
    
    for token in tokens :

        #replace slang words 
        if(slangDictionary.isSlang(token)==True):
            token=slangDictionary.replaceSlang(token)
            
        #if a letter exist in the dictionary replace it 
        token = ''.join([c if c not in d else d[c] for c in token ])
        
        #squeeze elongated tokens
        token=pattern.sub(r"\1\1", token)

        #if a token doesnot exist in the dictionary replace it 
        if(dictionary.check(token)==False and len(dictionary.suggest(token))>0):
            token=dictionary.suggest(token)[0]
        
        process+=token
        process+=" "

    process+="\n"

    return process


