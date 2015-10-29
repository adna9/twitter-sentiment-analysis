#import preProcess
#from lexicons import Slang
#import enchant

#read .tsv file and return labels and messages as lists
def opentsv(filepath):

    labels=[]
    messages=[]
    #process_messages=[]
    
    #load Slang Dictionary
    #slangDictionary = Slang.Slang()
    #load general purpose dictionary
    #dictionary = enchant.Dict("en_US")
    
    data = open(filepath,'r')


    #i=1
    for line in data.readlines():
        #print i
        #i+=1
        
        line=unicode(line,"utf-8")
        l = len(line.split("\t"))
        if l==4:
            labels.append(line.split("\t")[2])
            message=line.split("\t")[3]
            
            #processMessage=preProcess.processMessage(message,slangDictionary,dictionary)
            messages.append(message)
            #process_messages.append(processMessage)
        elif l==3:
            labels.append(line.split("\t")[1])
            message=line.split("\t")[3]
            messages.append(message)
            #processMessage=preProcess.processMessage(message,slangDictionary,dictionary)
            #process_messages.append(processMessage)
            

    data.close()

    #return labels,messages,process_messages
    return labels,messages

#read .tsv file ignoring neutral messages
def opentsvPolarity(filepath):

    labels=[]
    messages=[]
    #process_messages=[]
    
    #load Slang Dictionary
    #slangDictionary = Slang.Slang()
    #dictionary = enchant.Dict("en_US")
    data = open(filepath,'r')
  
    for line in data :
        line=unicode(line,"utf-8")
        #ignore neutral messages
        category = line.split("\t")[2]
        if(category != 'neutral') :
            labels.append(category)
            message=line.split("\t")[3]
            #processMessage=preProcess.processMessage(message,slangDictionary,dictionary)
            messages.append(message)
            #process_messages.append(processMessage)

    data.close()

    #return labels,messages,process_messages
    return labels,messages
