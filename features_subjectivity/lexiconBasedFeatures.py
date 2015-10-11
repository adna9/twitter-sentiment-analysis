def sumOfScores(lexicon,message,tokens,pos):
    if lexicon.__module__ == "lexicons.afinn.afinn":
        return lexicon.score(message)
    elif lexicon.__module__ == "lexicons.SentiWordNetLexicon":
        return lexicon.score(tokens,pos)
    else:
        return lexicon.score(tokens)

def maxOfScores(lexicon,tokens,pos):
    if lexicon.__module__ == "lexicons.SentiWordNetLexicon":
        max_score = lexicon.score(tokens[0],pos[0])
    else:
        max_score = lexicon.score(tokens[0])

    for i in range(1,len(tokens)):
        if lexicon.__module__ == "lexicons.SentiWordNetLexicon":
            x = lexicon.score(tokens[i],pos[i])
        else:
            x = lexicon.score(tokens[i])

        if x > max_score:
            max_score = x


    return max_score

def minOfScores(lexicon,tokens,pos):
    if lexicon.__module__ == "lexicons.SentiWordNetLexicon":
        min_score = lexicon.score(tokens[0],pos[0])
    else:
        min_score = lexicon.score(tokens[0])

    for i in range(1,len(tokens)):
        if lexicon.__module__ == "lexicons.SentiWordNetLexicon":
            x = lexicon.score(tokens[i],pos[i])
        else:
            x = lexicon.score(tokens[i])

        if x < min_score:
            min_score = x


    return min_score
