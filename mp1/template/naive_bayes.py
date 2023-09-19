
# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''


def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")


"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""


def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(
        trainingdir, testdir, stemming, lowercase, silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""


def bagOfWords(train_set, train_labels):
    # use counter to create a bag of words map
    positive_reviews = []
    negative_reviews = []
    for i in range(0, len(train_labels)):

        if (train_labels[i] == 1):
            positive_reviews += train_set[i]
        else:
            negative_reviews += train_set[i]
    positive = Counter(positive_reviews)
    negative = Counter(negative_reviews)
    return (positive, negative)

# 1. we know which reivews are pos and neg
# 2. we know how many words are in the pos and neg
# 3. we know how many of each word are in pos and neg

#dev

# 1. we do not know which reviews are pos and neg
# 2. we do know how many words are in the pos and neg
# 3. we do know how many of each word are in pos and neg

# we want to find which are pos and neg

def naiveBayes(dev_set, train_set, train_labels, laplace=1.0, pos_prior=0.9, silently=True):
    dev_labels = []
    positive, negative = bagOfWords(train_set,train_labels)
    neg_prior = 1.0 - pos_prior
    pos_words = sum(positive.values())
    neg_words = sum(negative.values())
    #Goal is to calculate the Probabilty of positive and negative given a review in the data set
    # we will do this buy doing P(pos|words) where words represent the words of a devset review set.
    # from that we will populate the dev_lables
    #  P(Type=Pos|Words) = P(Pos) ∏ All wordsP(Word|Pos)
    #  P(Type=Pos|Words) = P (pos) ∏  All wordsP(Word|Pos) the sum of all words in that review / total pos words
    
    print(pos_prior)
    print(math.log(pos_prior))
    
    for i in range (0,len(dev_set)):
        #dev_set[i] == the review
        counter = Counter(dev_set[i]) # we have hte count of each word in the review
        #traverese through this counter 
        pos_log_sum =0
        neg_log_sum =0
        for key,value in counter.items():
            if (key in positive):
                pos_log_sum +=  value* math.log( (positive[key] + laplace) / (pos_words + laplace *(1 + len(positive)+len(negative)))) 
            else:
                pos_log_sum +=  value* math.log(laplace / (pos_words + laplace *(1 + len(positive)+len(negative))) ) 
            if (key in negative):
                neg_log_sum +=  value* math.log( (negative[key] + laplace )/ (neg_words + laplace *(1 + len(positive)+len(negative))))    
            else:
                neg_log_sum += value* math.log(laplace / (neg_words + laplace *(1 + len(positive)+len(negative))))    
                
        pos_prob = math.log(pos_prior) + pos_log_sum
        neg_prob = math.log(neg_prior) + neg_log_sum
      
        if pos_prob > neg_prob:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    print(pos_words,neg_words)
    return dev_labels
   