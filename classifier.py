#!/home/clare/anaconda3/bin/python3

import os
import csv
from sklearn import svm
import numpy as np
import nltk
import re


path = "It-Bank/ACLData"
nltk.download('punkt')

testpath = "It-Bank/DevData"

def read_in_ACLData(path): #you can change the path, but this will read in all files in a folder. It returns 3 lists, one for each column
    answers = []
    positions = []
    sentences = []
    for filename in os.listdir(path):
        with open(path+"/"+filename) as data:
            text =  csv.reader(data,delimiter="\t")
            print("text:",text)
            for row in text:
                answers.append(row[0])
                positions.append(row[1])
                sentences.append(row[2])
    return answers, positions, sentences
    


def extract_wrd_bigrams(sentences, positions): 
    i = 0
    all_before_words = [] #list of words preceding it. I was going to make it a tuple, but the second part would always be "it" so that seemed silly
    all_after_words = []
    while i < len(sentences): #this while loop finds all the different words before and after eacn instance of "it"
        posn = int(positions[i])
        sent = nltk.word_tokenize(sentences[i])
        if posn > 1 and sent[(posn-2)] not in all_before_words: 
            all_before_words.append(sent[(posn-2)])
        if posn < len(sent) and sent[(posn)] not in all_after_words:
            all_after_words.append(sent[(posn)])
        i += 1
    wrd_array = np.zeros([len(sentences), (len(all_after_words)+len(all_before_words))])

    j = 0
    while j < len(sentences): #this while loop finds all the different words before and after eacn instance of "it"
        posn = int(positions[j]) - 1
        sent = nltk.word_tokenize(sentences[j])
        for k in range(len(sent)):
            for word in all_before_words:
                if posn > 0 and word == sent[posn-1]:
                    wrd_array[j][k] = 1
            for word in all_after_words:
                if posn < len(sent) -2 and word == sent[posn+1]:
                    wrd_array[j][k] = 1
        j += 1
    return wrd_array
           
answers, positions, sentences = read_in_ACLData(path)
wrd_array = extract_wrd_bigrams(sentences, positions) #this is very sparse
#print(wrd_bg_ft[0])
#print(len(answers)) #these are just little check-ins. change as needed
#print(len(positions))
#print(len(sentences))

#extracting testing data
testanswers,testpositions,testsentences = read_in_ACLData(path)
test_array = extract_wrd_bigrams(testsentences,testpositions)
Z = np.array(testanswers)

Y = np.array(answers) #This np array is now ready to be used in the classifier. That's all we need to do to it
clf = svm.SVC()
clf.fit(wrd_array, Y)
print(clf.score(test_array,Z))


