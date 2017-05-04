#!/home/clare/anaconda3/bin/python3

import os
import csv
import string
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
path = "It-Bank/ACLData"
#nltk.download('punkt')

testpath = "It-Bank/DevData"

cues = [''] #uncomment this and comment the next line to "switch off" cue words as features
#cues = ['seems','rain','rains','raining','rained','snow','snows','snowing','snowed','important','necessary']

def read_in_ACLData(path):  # you can change the path, but this will read in all files in a folder. It returns 3 lists, one for each column
    answers = []
    positions = []
    sentences = []
    for filename in os.listdir(path):
        with open(path+"/"+filename) as data:
            text = csv.reader(data, delimiter="\t")
            for row in text:
                answers.append(row[0])
                positions.append(row[1])
                sentences.append(row[2])
    return answers, positions, sentences
    


def extract_wrd_bigrams(sentences, positions): 
    i = 0
    all_before_words = []  # list of words preceding it. I was going to make it a tuple, but the second part would always be "it" so that seemed silly
    all_after_words = []
    two_before = []
    two_after = []
    while i < len(sentences):  # this while loop finds all the different words before and after eacn instance of "it"
        posn = int(positions[i])
        sent = sentences[i].lower()
        sent = sent.split(" ")
        sent[0] = 'BEGIN'
        sent.append('END')
        #print(sent)
        #print()
        #print(" posn: " + sent[posn]) #I put this here just in case
        if lemmatizer.lemmatize(sent[posn-1]) not in all_before_words:
            word_before = lemmatizer.lemmatize(sent[posn-1])
            all_before_words.append(word_before)
        if lemmatizer.lemmatize(sent[posn+1]) not in all_after_words:
            word_after = lemmatizer.lemmatize(sent[posn+1])
            all_after_words.append(word_after)
        if posn > 2 and lemmatizer.lemmatize(sent[posn-2]) not in all_before_words:
            word_before = lemmatizer.lemmatize(sent[posn-2])
            all_before_words.append(word_before)
        if posn < len(sent)-3 and lemmatizer.lemmatize(sent[posn+2]) not in all_after_words:
            word_after = lemmatizer.lemmatize(sent[posn+2])
            all_after_words.append(word_after)


			
        i += 1
    return all_before_words,all_after_words,two_before,two_after


def extract_POS_bigrams(sentences, positions):
    i = 0
    all_before_POS = []
    all_after_POS = []
    while i < len(sentences):
        posn = int(positions[i])
        sent = sentences[i].lower()
        sent = sent.split(" ")
        sent[0] = 'BEGIN'
        sent.append('END')
        sent_POS = nltk.pos_tag(sent)
        if posn > 1 and sent_POS[(posn-1)][1] not in all_before_POS:
            all_before_POS.append(sent_POS[(posn-1)][1])
        if posn < len(sent_POS)-2 and sent_POS[(posn+1)][1] not in all_after_POS:
            all_after_POS.append(sent_POS[posn+1][1])
        i += 1
    return all_before_POS, all_after_POS

def get_feat_vect(all_before_words, all_after_words, two_before, two_after, before_POS, after_POS, sentences, positions):
    j = 0
    wrd_array = np.zeros([len(sentences), (len(all_before_words)+len(all_after_words)+len(two_before)+len(two_after)+len(before_POS)+len(after_POS)+2)])
    print(wrd_array.shape)
    while j < len(sentences):  # this while loop finds all the different words before and after each instance of "it"
        posn = int(positions[j])
        sent = sentences[j].lower()
        sent = sent.split(" ")
        sent[0] = 'BEGIN'
        sent.append('END')
        sent_POS = nltk.pos_tag(sent)
        
        wrd_bf = sent[posn-1]
        wrd_af = sent[posn+1]
        if posn > 1: 
            pos_bf = sent_POS[posn-1][1]
        if posn < len(sent) -2:
            pos_af = sent_POS[posn+1][1]
        if posn > 2:
            twobf = sent[posn-2]
        if posn < len(sent) - 3:
            twoaf = sent[posn+2]
        if wrd_bf in all_before_words:
            wrd_array[j][all_before_words.index(wrd_bf)] = 1
        if wrd_af in all_after_words:
            wrd_array[j][(len(all_before_words)+all_after_words.index(wrd_af))] = 1
        if pos_bf in before_POS:
            wrd_array[j][(len(all_before_words)+len(all_after_words)+before_POS.index(pos_bf))] = 1
        if pos_af in after_POS:
            wrd_array[j][(len(all_before_words)+len(all_after_words)+len(before_POS)+after_POS.index(pos_af))] = 1
        wrd_array[j][(len(all_before_words)+len(all_after_words)+len(before_POS)+len(after_POS))] = posn
        wrd_array[j][(len(all_before_words)+len(all_after_words)+len(before_POS)+len(after_POS))+1] = len(sent)
        if twobf in two_before:
            wrd_array[j][(len(all_before_words)+len(all_after_words)+len(before_POS)+len(after_POS))+2+two_before.index(twobf)] = 1
        if twoaf in two_after:
            wrd_array[j][(len(all_before_words)+len(all_after_words)+len(before_POS)+len(after_POS))+2+len(two_before)+two_after.index(twoaf)] = 1
        j += 1
    return wrd_array
           
answers, positions, sentences = read_in_ACLData(path) # self-explanatory
#print(len(answers))
before_words,after_words, two_before, two_after = extract_wrd_bigrams(sentences, positions) # get bag (bags) of words
before_POS, after_POS = extract_POS_bigrams(sentences, positions)
#print(before_words)
#print(after_words)
feature_vector = get_feat_vect(before_words,after_words,two_before, two_after,before_POS, after_POS, sentences, positions) # use bag of words and sentences to get feature vectors
# print(wrd_bg_ft[0])
# print(len(answers)) #these are just little check-ins. change as needed
# print(len(positions))
# print(len(sentences))

# extracting testing data
#testanswers,testpositions,testsentences = read_in_ACLData(testpath)
# don't need to get Bag Of Words, we're using the training bag against the test sentences
#test_feature_vector = get_feat_vect(before_words, after_words, testsentences, testpositions)  # use training bag of words and test sentences to get feature vectors
#Z = np.array(testanswers)

Y = np.array(answers)  # This np array is now ready to be used in the classifier. That's all we need to do to it
clf = svm.LinearSVC()
clf.fit(feature_vector[:1700], Y[:1700])
print(clf.predict(feature_vector[1700:]))


# Scoring
print(clf.score(feature_vector[1700:], Y[1700:]))
print(precision_score(clf.predict(feature_vector), Y, labels=['0', '1'], average='macro'))
print(recall_score(clf.predict(feature_vector), Y, labels=['0', '1'], average='macro'))
print(f1_score(clf.predict(feature_vector), Y, labels=['0', '1'], average='macro'))



