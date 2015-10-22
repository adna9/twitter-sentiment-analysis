#check if a word of the message appears in a cluster
def checkClusters(tokens,clusters):
##    tags = []
##
##    for cluster_words in clusters.d.values():
##        l = [x for x in tokens if x in cluster_words]
##        if len(l)>0:
##            tags.append(1)
##        else:
##            tags.append(0)
##
    
    #initialize list with zeros
    tags = [0] * len(clusters.keys)

    c = []

    for token in tokens:
        c.append(clusters.d.get(token,"no_cluster"))

    c = [x for x in c if x!="no_cluster"] 

    for i in c:
        tags[clusters.keys.index(i)] = 1
    
    return tags
