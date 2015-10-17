#check if a message has negation
def hasNegation(tokens,negationList):
    for token in tokens:
        if token in negationList:
            return 1

    return 0
