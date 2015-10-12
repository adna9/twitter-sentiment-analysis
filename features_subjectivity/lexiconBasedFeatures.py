def sumOfScores(lexicon,message,tokens,pos):
    if lexicon.__module__ == "lexicons.afinn.afinn":
        return lexicon.score(message)
    elif lexicon.__module__ == "lexicons.SentiWordNetLexicon":
        return lexicon.score(tokens,pos)
    else:
        return lexicon.score(tokens)


def maxOfScores(lexicon,tokens,pos):

    #we use absolute values because both positive and negative values
    #are considered subjective
    
    if lexicon.__module__ == "lexicons.SentiWordNetLexicon":
        max_score = abs(lexicon.score(tokens[0],pos[0]))
    else:
        max_score = abs(lexicon.score(tokens[0]))

    for i in range(1,len(tokens)):
        if lexicon.__module__ == "lexicons.SentiWordNetLexicon":
            x = abs(lexicon.score(tokens[i],pos[i]))
        else:
            x = abs(lexicon.score(tokens[i]))

        if x > max_score:
            max_score = x


    return max_score

def minOfScores(lexicon,tokens,pos):
    
    #we use absolute values because both positive and negative values
    #are considered subjective
    
    if lexicon.__module__ == "lexicons.SentiWordNetLexicon":
        min_score = abs(lexicon.score(tokens[0],pos[0]))
    else:
        min_score = abs(lexicon.score(tokens[0]))

    for i in range(1,len(tokens)):
        if lexicon.__module__ == "lexicons.SentiWordNetLexicon":
            x = abs(lexicon.score(tokens[i],pos[i]))
        else:
            x = abs(lexicon.score(tokens[i]))

        if x < min_score:
            min_score = x


    return min_score

def numberOfAppearances(lexicon,tokens):

    total = 0
    
    if lexicon.__module__ == "lexicons.afinn.afinn":
        for token in tokens:
            if len(lexicon.find_all(token)) > 0:
                total+=1
    else:
        total = lexicon.getNumberOfAppearances(tokens)
        


    return total

def scoreOfLastWord(lexicon,lastToken,lastPosTag):
    if lexicon.__module__ == "lexicons.SentiWordNetLexicon":
        return lexicon.score(lastToken,lastPosTag)
    else:
        return lexicon.score(lastToken)

def scoreOfLastWordAppearedInLexicon(lexicon,tokens,pos):
    #iterate tokens from the end
    #if token is in lexicon then break
    for i in range(len(tokens)-1,-1,-1):
        if numberOfAppearances(lexicon,tokens[i]) > 0:
            return sumOfScores(lexicon,tokens[i],tokens[i],pos[i])

    return 0
