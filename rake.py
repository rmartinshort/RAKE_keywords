#!/usr/bin/env python 
#RMS 2019
# -*- coding: utf-8 -*-


"""
Classes for the RAKE algorithm
As described in the paper `Automatic keyword extraction from individual
documents` by Stuart Rose, Dave Engel, Nick Cramer and Wendy Cowley (2010).

This code was heavily influenced by 
https://github.com/csurfer/rake-nltk
"""

import string 
import nltk.tokenize as nlptok
import nltk
from itertools import chain, groupby, product
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


class RakeSummary(object):

    def __init__(self,language='english',stopwords=None,additional_stopwords=[],punctuation=None,min_ngram=1,max_ngram=10):

        """
        Wrapper for RAKE when we want to run it on a datafram and report corpus level metrics
        """

        self.rakeobj = Rake(language=language,stopwords=stopwords,\
            additional_stopwords=additional_stopwords,\
            punctuation=punctuation,min_ngram=min_ngram,\
            max_ngram=max_ngram)

    def run_rake(self,documents):

        """
        Run RAKE on each (possibly preprocessed) document in the input string documents
        """

        extracted = []
        candidate = []

        i = 0

        for doc in documents:

                self.rakeobj.determine_keywords(doc,docID=i)
                extracted.append(self.rakeobj.return_extracted_phrases())
                candidate.append(self.rakeobj.return_candidate_phrases())
                i += 1

        extracted = pd.concat(extracted)
        candidate = pd.concat(candidate)
        extracted.columns = ['phrase','extracted_score','docID']
        candidate.columns = ['phrase','candidate_score','docID']


        #Count the number of occurences of each phrase in the entire corpus
        extracted_counts = extracted[['phrase','extracted_score']].groupby('phrase').count()
        candidate_counts = candidate[['phrase','candidate_score']].groupby('phrase').count()

        #join the dataframes
        overall_counts = pd.merge(extracted_counts,candidate_counts,how='outer',left_index=True,right_index=True).fillna(0)
        overall_counts.columns = ['extracted_count','candidate_count']

        ### Determine the corpus level metrics 'essentiality' and 'exclusivity' for each phrase. These are defined in 
        ### Rose et al. (2010)

        overall_counts['exclusivity'] = overall_counts['extracted_count'].div(overall_counts['candidate_count'])
        overall_counts['essentiality'] = overall_counts['candidate_count'].mul(overall_counts['exclusivity'])

        return overall_counts.reset_index()
    


class Rake(object):

    def __init__(self,language='english',stopwords=None,punctuation=None,additional_stopwords=[],min_ngram=1,max_ngram=10):

        """
        RAKE algorithm allowing additional stopwords and language support for whatever languages 
        the nltk package supports
        """

        if stopwords == None:
            stopwords = nltk.corpus.stopwords.words(language)
            self.stopwords = stopwords + additional_stopwords
        else:
            self.stopwords = stopwords + additional_stopwords

        if punctuation == None:
            self.punctuation = string.punctuation
        else:
            self.punctuation = punctuation 

        self.chars_to_ignore = set(chain(self.stopwords,self.punctuation))

        self.min_ngram = min_ngram
        self.max_ngram = max_ngram

        self.degree = None 
        self.frequency = None
        self.candidate_phrases = None
        self.extracted_phrases = None 
        self.phrase_list = None

    def determine_keywords(self,text,docID):

        """
        User calls this function to run the RAKE algorithm on a single document
        """

        self._get_candidate_phrases(text)
        self._get_frequency()
        self._get_degree()

        T = len(self.degree)
        len2extract = self._len_extracted_phrases(T)

        self._score_phrases(docID,len2extract)


    def return_phrase_list(self):

        return self.phrase_list

    def return_frequency(self):
    
        return self.frequency 

    def rerurn_degree(self):

        return self.degree

    def return_extracted_phrases(self):

        return self.extracted_df

    def return_candidate_phrases(self):

        return self.candidate_df


    ### Internal functions


    def _len_extracted_phrases(self,T,rule='OneThird'):

        if rule == 'OneThird':

            return T//3

    def _get_frequency(self):

        self.frequency = Counter(chain.from_iterable(self.phrase_list))

    def _get_degree(self):
    
        co_occurence_graph = defaultdict(lambda: defaultdict(lambda: 0))
        
        for phrase in self.phrase_list:
                    
            #This gets all combinations of the words inthe phrase
            for (word,coword) in product(phrase,phrase):
                co_occurence_graph[word][coword] += 1
        
        
        #The degree of each word is the number of co-occurences it has
        #The freqeuncy of each word is the number of times it appears in each
        #sentence
        
        self.degree = defaultdict(lambda: 0)
        for key in co_occurence_graph.keys():
            self.degree[key] = sum(co_occurence_graph[key].values())
            

    def _score_phrases(self,docID,T):

        L = len(self.phrase_list)
    
        rank_list = [None]*L
        phrases = [None]*L
        
        i = 0
        for phrase in self.phrase_list:
            score = 0
            for word in phrase:
                score += self.degree[word]/self.frequency[word]
            rank_list[i] = score
            phrases[i] = " ".join(phrase)
            i += 1
            
        self.candidate_df = pd.DataFrame({'phrase':phrases,'score':rank_list,'documentID':docID}).\
        sort_values(by='score',ascending=False).reset_index(drop=True)
        self.extracted_df = self.candidate_df.iloc[:T]
        

    def _get_candidate_phrases(self,text):
    
        sentences = nlptok.sent_tokenize(text)
        self.phrase_list = set()
        
        for sent in sentences:
            #Split into all words and punctuation
            word_list = [word.lower() for word in nlptok.wordpunct_tokenize(sent)]
            #print(word_list)

            #split by stop words and group
            groups = groupby(word_list,lambda x: x not in self.chars_to_ignore)

            phrases = [tuple(group[1]) for group in groups if group[0]==True]

            #filter according to n-gram criteria
            filtered_phrases = list(filter(lambda x: self.min_ngram <= len(x) <= self.max_ngram, phrases))
            
            self.phrase_list.update(filtered_phrases)









