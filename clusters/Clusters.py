#class for the twitter word clusters
class Clusters():

    #clusters directory
    directory = "clusters/TwitterWordClusters/"

    #clusters file
    file1 = "50mpaths2.sdx"

    #constructor
    def __init__(self):
        #initialize dictionary, dictionary represents clusters
        # d [cluster_id] = word_list 
        self.d = {}

        #load cluster
        self.loadClusters()

    #load Clusters
    def loadClusters(self):
        #open file
        f = open(Clusters.directory+Clusters.file1,"r")
        
        for line in f.readlines():
            cluster_id = line.split("\t")[0]
            word = line.split("\t")[1]
            
            if cluster_id not in self.d.keys():
                #new entry
                self.d[cluster_id] = []

            #add new word
            self.d[cluster_id].append(word)

        f.close()
