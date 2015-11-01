#return avg,min,max scores
def F1PosTagsScore(pos,pos_tags_scores):
    scores = []

    for x in pos:
        scores.append(pos_tags_scores.get(x,0))
        
    average = sum(scores)/float(len(scores))
    maximum = max(scores)
    minimum = min(scores)

    return average, maximum, minimum

def F1PosBigramsScore(pos,pos_bigrams_scores):
    scores = []

    for x in pos:
        scores.append(pos_bigrams_scores.get(x,0))
        
    average = sum(scores)/float(len(scores))
    maximum = max(scores)
    minimum = min(scores)

    return average, maximum, minimum

#return avg,min,max scores
def F1PosTrigramsScore(pos,pos_trigrams_scores):
    scores = []

    for x in pos:
        scores.append(pos_trigrams_scores.get(x,0))
        
    average = sum(scores)/float(len(scores))
    maximum = max(scores)
    minimum = min(scores)

    return average, maximum, minimum
