# bigram_naive_bayes.py
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
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def bagOfWords(train_set, train_labels):
    # use counter to create a bag of words map
    positive_reviews = []
    bigramP =[]
    negative_reviews = []
    bigramN =[]
    for i in range(0, len(train_labels)):

        if (train_labels[i] == 1):
            positive_reviews += train_set[i] #add a single reivew into the array of reivews
            for b in range(0,len(train_set[i])-1):
                bigram =(train_set[i][b],train_set[i][b+1])
                bigramP.append(bigram)
        else:
            negative_reviews += train_set[i] #add a single reivew into the array of reivews
            for b in range(0,len(train_set[i])-1):
                bigram =(train_set[i][b],train_set[i][b+1])
                bigramN.append(bigram)
            
    positive = Counter(positive_reviews)
    negative = Counter(negative_reviews)
    positive_bigram = Counter(bigramP) 
    negative_bigram = Counter(bigramN)
    return (positive, negative, positive_bigram, negative_bigram)


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigramBayes(dev_set, train_set, train_labels, unigram_laplace=0.8, bigram_laplace=0.6, bigram_lambda=0.5, pos_prior=0.3, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    dev_labels = []
    positive, negative, positive_bigram, negative_bigram = bagOfWords(train_set,train_labels)
    neg_prior = 1.0 - pos_prior
    pos_words = sum(positive.values())
    neg_words = sum(negative.values())
    pos_pairs = sum(positive_bigram.values())
    neg_pairs = sum(negative_bigram.values())
    #print(positive_bigram)
    
    for i in range (0,len(dev_set)):
        #dev_set[i] == the review
        unigramCounter = Counter(dev_set[i]) # this is the counter for a single review in the developmetn set
                                      # it will get us the count of each unique word
        bigram_array =[];                                      
        for b in range(0,len(dev_set[i]) - 1):
            bigram = (dev_set[i][b], dev_set[i][b + 1])
            bigram_array.append(bigram)    #this will give us the bigram array of the dev set review        
            
        bigramCounter = Counter(bigram_array) #this will give us the bigram counter of the devset review
        pos_log_sum =0
        neg_log_sum =0
        seen_count =0
        seen_count_bigram =0
        
        for key,value in unigramCounter.items(): #mnight not need this loop
            if (key in positive or key in negative): #see how many words in the dev set appear in the positive or negative of the training set
               seen_count +=1 
        for key,value in unigramCounter.items():
            if (key in positive):
                pos_log_sum +=   value * math.log( (positive[key] + unigram_laplace) / (pos_words + unigram_laplace *(1 + len(positive)+len(negative)))) 
                # We first are doing unigram_laplace by getting the probability of a word and adding the alpha / total words
            else:
                pos_log_sum +=   value * math.log(unigram_laplace / (pos_words + unigram_laplace *(1 + len(positive)+len(negative))) ) 
            if (key in negative):
                neg_log_sum +=   value * math.log( (negative[key] + unigram_laplace )/ (neg_words + unigram_laplace *(1 + len(positive)+len(negative))))    
            else:
                neg_log_sum +=  value * math.log(unigram_laplace / (neg_words + unigram_laplace *(1 + len(positive)+len(negative))))   
                 
        Ppos_unigram = (1-bigram_lambda) * (math.log(pos_prior) + pos_log_sum)
        Pneg_unigram = (1-bigram_lambda) * (math.log(neg_prior) + neg_log_sum)
        
        pos_log_sum_bigram =0
        neg_log_sum_bigram =0
        for key,value in bigramCounter.items():
            if (key in positive_bigram or key in negative_bigram):
                seen_count_bigram +=1 
        for key,value in bigramCounter.items():
            if (key in positive_bigram):
                pos_log_sum_bigram +=   value * math.log( (positive_bigram[key] + bigram_laplace) / (pos_pairs + bigram_laplace *(1 + len(positive_bigram)+len(negative_bigram)))) 
                # We first are doing bigram_laplace by getting the probability of a word and adding the alpha / total words
            else:
                pos_log_sum_bigram +=   value * math.log(bigram_laplace / (pos_pairs + bigram_laplace *(1 + len(positive_bigram)+len(negative_bigram))) ) 
            if (key in negative_bigram):
                neg_log_sum_bigram +=   value * math.log( (negative_bigram[key] + bigram_laplace )/ (neg_pairs + bigram_laplace *(1 + len(positive_bigram)+len(negative_bigram))))    
            else:
                neg_log_sum_bigram +=  value * math.log(bigram_laplace / (neg_pairs + bigram_laplace *(1 + len(positive_bigram)+len(negative_bigram))))   
                
        Ppos_bigram = (bigram_lambda) * (math.log(pos_prior) + pos_log_sum_bigram)
        Pneg_bigram = (bigram_lambda) * (math.log(neg_prior) + neg_log_sum_bigram)
        
        pos_prob = Ppos_bigram +Ppos_unigram
        neg_prob = Pneg_bigram + Pneg_unigram
        
        if pos_prob > neg_prob:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    return dev_labels


'''
laplace=  word + alpha/ total + alpha(seen+1)
naive bayes = product of all of htese probabilites * the probability of the prior we can makesure underflow doesn't happen by applying log

log(P(prior))  + Summation of each log(laplace)
'''


