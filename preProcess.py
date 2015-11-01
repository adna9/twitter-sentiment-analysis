import re

def processMessage(messages,tokens,pos,slangDictionary,dictionary,pattern,d,tags):

    #the processed message
    process=""

    for i in range(0,len(pos)):
        token = tokens[i]
        tag = pos[i]

        #replace slang words 
        if(slangDictionary.isSlang(token)):
            token=slangDictionary.replaceSlang(token)
            
        #if a letter exist in the dictionary replace it 
        token = ''.join([c if c not in d else d[c] for c in token ])
        
        #squeeze elongated tokens
        token=pattern.sub(r"\1\1", token)

        # very very slow 
        #if a token doesnot exist in the dictionary replace it 
        #if(dictionary.check(token)==False and len(dictionary.suggest(token))>0):
        #    token=dictionary.suggest(token)[0]
        if tag not in tags:
            if not dictionary.check(token):
                try:
                    token = dictionary.suggest(token)[0]
                except:
                    pass
        
                
            
        process+=token
        process+=" "

    process+="\n"

    return process

def preprocessMessages(messages,tokens,pos,slangDictionary,dictionary):
    processed_messages = []
    tags = ["U","E","$",","]

    
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
    
    
    for i in range(0,len(messages)):
        processed_messages.append(processMessage(messages[i],tokens[i],pos[i],slangDictionary,dictionary,pattern,d,tags))

    return processed_messages
