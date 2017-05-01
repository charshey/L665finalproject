import os
from sklearn import svm
import numpy as np
import nltk
import re


path = "C:/Users/ihovl/Documents/applied ml/It-Bank/It-Bank/ACLData/"


def read_in_ACLData(path): #you can change the path, but this will read in all files in a folder. It returns 3 lists, one for each column
    answers = []
    positions = []
    sentences = []
    for filename in os.listdir(path):
        f = open(path+filename)
        line = f.readline()
        while line:
            rline = line.rstrip()
            splitted = rline.split("\t")
            answers.append(splitted[0])
            positions.append(splitted[1])
            sentences.append(splitted[2])
            line = f.readline()
    return answers, positions, sentences
    
    

def extract_wrd_bigrams(sentences, positions): 
    i = 0
    all_before_words = [] #list of words preceding it. I was going to make it a tuple, but the second part would always be "it" so that seemed silly
    all_after_words = []
    while i < len(sentences): #this while loop finds all the different words before and after eacn instance of "it"
        posn = int(positions[i])
        sent = sentences[i]
        sentsplit = sent.split(' ')
        if sentsplit[(posn-1)] not in all_before_words: 
            all_before_words.append(sentsplit[(posn-1)]) 
        if sentsplit[(posn+1)] not in all_after_words:
            all_after_words.append(sentsplit[(posn+1)])
        i += 1
    wrd_array = np.zeros([len(sentences), (len(all_after_words)+len(all_before_words))])

    j = 0
    while j < len(sentences): #this while loop finds all the different words before and after eacn instance of "it"
        posn = int(positions[j])
        sent = sentences[j]
        sentsplit = sent.split(' ')
        wrd_array[j][all_before_words.index(sentsplit[posn-1])] = 1
        wrd_array[j][(len(all_before_words)+all_after_words.index(sentsplit[posn+1]))] = 1
        j += 1
    return wrd_array
        
    
        



    
    
answers, positions, sentences = read_in_ACLData(path)
wrd_array = extract_wrd_bigrams(sentences, positions) #this is very sparse
#print(wrd_bg_ft[0])
#print(len(answers)) #these are just little check-ins. change as needed
#print(len(positions))
#print(len(sentences))

Y = np.array(answers) #This np array is now ready to be used in the classifier. That's all we need to do to it
clf = svm.SVC()
clf.fit(wrd_array[:1700], Y[:1700])
print(clf.predict(wrd_array[1700:]))


