def loadNegations():
    directory = "lexicons/negationsList/"
    file1 = "negations.txt"

    f = open(directory+file1,"r")

    negationsList = []

    for line in f.readlines():
        negationsList.append(line[0:len(line)-1])


    f.close()

    return negationsList
