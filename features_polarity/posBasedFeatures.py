
#calculate the number of adjectives in the message
def numberOfAdjectives(pos):
    return len([x for x in pos if x=="A"])

#calculate the number of adverbs
def numberOfAdverbs(pos):
    return len([x for x in pos if x=="R"])

#calculate the number of interjections
def numberOfIntejections(pos):
    return len([x for x in pos if x=="!"])

#calculate the number of verbs
def numberOfVerbs(pos):
    return len([x for x in pos if x=="V"])

#calculate the number of nouns
def numberOfNouns(pos):
    return len([x for x in pos if x=="N"])

#calculate the number of proper nouns
def numberOfProperNouns(pos,tokens):
    x = 0

    for i in range(0,len(pos)):
        #pos tagger wrongly tags these words as a proper noun
        if pos[i]=="^" and not(tokens[i]=="AT_USER" or tokens[i]=="EMOTICON_SAD" or tokens[i]=="EMOTICON_HAPPY" or tokens[i]=="HTTP_LINK"):
            x+=1

    return x
            


