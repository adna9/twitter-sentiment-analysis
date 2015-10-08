#read .tsv file and return labels and messages as lists
def opentsv(filepath):

    labels=[]
    messages=[]
    data = open(filepath,'r')

    for line in data.readlines():
        line=unicode(line,"utf-8")
        labels.append(line.split("\t")[2])
        messages.append(line.split("\t")[3])

    data.close()

    return labels,messages

#read .tsv file ignoring neutral messages
def opentsvPolarity(filepath):
    labels=[]
    messages=[]

    
    data = open(filepath,'r')
  
    for line in data :
        line=unicode(line,"utf-8")
        #ignore neutral messages
        category = line.split("\t")[2]
        if(category != 'neutral') :
            labels.append(category)
            messages.append(line.split("\t")[3])

    data.close()

    return labels,messages
