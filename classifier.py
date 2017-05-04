#!/home/clare/anaconda3/bin/python3

import os
import csv
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
path = "It-Bank/ACLData"
#nltk.download('punkt')

testpath = "It-Bank/DevData"

cues_array =['']

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
    while i < len(sentences):  # this while loop finds all the different words before and after each instance of "it"
        posn = int(positions[i])
        sent = nltk.word_tokenize(sentences[i])
        if posn > 1 and sent[(posn-2)] not in all_before_words:
            word_before = lemmatizer.lemmatize(sent[posn-2])
            all_before_words.append(word_before)
        if posn < len(sent) and sent[posn] not in all_after_words:
            word_after = lemmatizer.lemmatize(sent[posn])
            all_after_words.append(word_after)
        i += 1
    return all_before_words,all_after_words


def extract_POS_bigrams(sentences, positions):
    i = 0
    all_before_POS = []
    all_after_POS = []
    while i < len(sentences):
        posn = int(positions[i])
        sent_tok = nltk.word_tokenize(sentences[i])
        sent_POS = nltk.pos_tag(sent_tok)
        if posn > 1 and sent_POS[(posn-2)][1] not in all_before_POS:
            all_before_POS.append(sent_POS[(posn-2)][1])
        if posn < len(sent_POS) and sent_POS[(posn)][1] not in all_after_POS:
            all_after_POS.append(sent_POS[posn][1])
        i += 1
    return all_before_POS, all_after_POS



def get_feat_vect(all_before_words, all_after_words, all_before_POS, all_after_POS, sentences, positions):
    j = 0
    wrd_array = np.zeros([len(sentences), (len(all_after_words)+len(all_before_words)+len(all_before_POS)+len(all_after_POS))])
    while j < len(sentences):  # this while loop finds all the different words before and after each instance of "it"
        posn = int(positions[j])
        sent = nltk.word_tokenize(sentences[j])
        for k in range(len(sent)):
            for word in all_before_words:
                if posn > 1 and word == sent[posn-2]:
                    wrd_array[j][k] = 1
            for word in all_after_words:
                if posn < len(sent) - 1 and word == sent[posn]:
                    wrd_array[j][k] = 1
            for pos in all_before_POS:
                if posn > 1 and pos == sent[posn-2]:
                    wrd_array[j][k] = 1
            for pos in all_after_POS:
                if posn < len(sent) - 1 and pos == sent[posn]:
                    wrd_array[j][k] = 1
        j += 1
    return wrd_array
           
answers, positions, sentences = read_in_ACLData(path) # self-explanatory
before_words, after_words = extract_wrd_bigrams(sentences, positions) # get bag (bags) of words

print(before_words)
print(after_words)

before_POS, after_POS = extract_POS_bigrams(sentences, positions)

print(before_POS)
print(after_POS)

feature_vector = get_feat_vect(before_words,after_words, before_POS, after_POS, sentences, positions) # use bag of words and sentences to get feature vectors
# print(wrd_bg_ft[0])
# print(len(answers)) #these are just little check-ins. change as needed
# print(len(positions))
# print(len(sentences))

# extracting testing data
testanswers, testpositions, testsentences = read_in_ACLData(testpath)
# don't need to get Bag Of Words, we're using the training bag against the test sentences
test_feature_vector = get_feat_vect(before_words, after_words, before_POS, after_POS, testsentences, testpositions)  # use training bag of words and test sentences to get feature vectors
Z = np.array(testanswers)

Y = np.array(answers)  # This np array is now ready to be used in the classifier. That's all we need to do to it
clf = svm.SVC(kernel='poyl', class_weight='balanced')
clf.fit(feature_vector, Y)
print(clf.predict(test_feature_vector))


# Scoring idk how precision and recall work soo figure that out later
print(clf.score(test_feature_vector, Z))
#print(precision_score(clf.predict(test_feature_vector), Z, labels=['0', '1'], average='micro'))
#print(recall_score(clf.predict(test_feature_vector), Z, labels=['0', '1'], average='micro'))
print(Z)


