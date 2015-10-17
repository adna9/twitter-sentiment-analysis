#check if a word of the message appears in a cluster
def checkClusters(tokens,clusters):
    tags = []

    for cluster_words in clusters.d.values():
        l = [x for x in tokens if x in cluster_words]
        if len(l)>0:
            tags.append(1)
        else:
            tags.append(0)
        
    return tags
