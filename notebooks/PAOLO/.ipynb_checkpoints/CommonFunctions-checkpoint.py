# Gensim
import gensim
import gensim.corpora as corpora
# from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pickle
import pandas as pd
import numpy as np
import math
import seaborn as sns
from wordcloud import WordCloud

# pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

from os import path

from Constants import *



#####
## from `ltc_patients` create `list_of_patients` if it does not exist
## return list of BOW, one per patient
## return list of unique disease terms
def create_bows(bin_matrix):
    list_of_patients = []


    patients = bin_matrix['patient_id'].unique()
    ltcs = bin_matrix.drop('patient_id', axis = 1)
    index = 0

    # Iterate through patients
    for patient in patients:

        # Start with empty list of LTCs for each patient
        patient_ltcs = []

        # Iterate through each LTC for patient
        for ltc in ltcs:

            # Check if patient has LTC
            if bin_matrix.at[index, ltc] == 1:

                # If LTC present, add to list of patient LTCs
                patient_ltcs.append(ltc)   

        # Add list of patient LTCs to list of patients        
        list_of_patients.append(patient_ltcs)

        # Increment index by 1
        index+=1

    ## cache for future use
    with open(BOWs, 'wb') as f:
        pickle.dump(list_of_patients, f)

    with open(LTCs, 'wb') as f:
            pickle.dump(ltcs, f)
            
        
    return list_of_patients, ltcs.columns







        
# Topics generation

# in: bow is the list of bag of words
# in: topics_count is the number of topics to be generated
# returns lda-model
## this method saves the model as a pickle file -- using topics_count as suffix to separate different topics configurations

def bagOfWords2Topics(bow, topics_count):
    id2word = corpora.Dictionary(bow)

    
    corpus = []
    for text in bow:
        new = id2word.doc2bow(text)
        corpus.append(new)

    lda_model =  gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=topics_count,
                                           workers=3,
                                           random_state=100,
                                           chunksize=500,
                                           passes=10,
                                           alpha="asymmetric",
                                           eta=1.0)
    ## cache for future use

    ## suffix the base file name
    LDA_MODEL_SUFF = path.dirname(LDA_MODEL)+ path.basename(LDA_MODEL).split('.')[0]+ "_"+ str(topics_count)+"."+path.basename(LDA_MODEL).split('.')[1]
    
    with open(LDA_MODEL_SUFF, 'wb') as f:
        pickle.dump(lda_model, f)
        print("model saved to: {0}".format(LDA_MODEL_SUFF))
        
    return lda_model


# compute the probabilities of each bow relative to each topic

def compute_all_bow_probabilities(lda_model):
    # Will store topic similarity score for each of the patients in our corpus
    lda_res_each_ptnt = []

    ## for each bag of words used to generate the topics:
    for i in bows:
        bow = id2word.doc2bow(i)
        lda_res_each_ptnt.append(lda_model[bow])  ## these are the probabilities for this document for each topic
        
    return lda_res_each_ptnt


# input: list of disease terms
# input: corpus_size = number of bows
# return list of ids ordered by the same order as the input terms
def calculate_idf(terms, corpus_size, term_occur):
    
    idf = list()
    for i in range (len(terms)):
        idf.append(math.log10(corpus_size / (term_occur[terms[i]] + 1)))
    return idf


## return a dict term:<number of occurrences>
def term_occurrences(bows):
    
    term_occur = dict()    
    for bow in bows:
        for t in bow:
            try:
                term_occur[t] = term_occur[t] + 1
            except:
                term_occur[t] = 1
    return term_occur


# Build a dataframe with each term's relative weight within each topic
# in: lda_model

## input:
## LDA model computed in the previous step
## number of topics in the model

## 1 get terms  native weights from lda_model: weight(term, topic)
## 2 calculate idf for each term in the corpus idf(term)
## to eeach term in topic assign weight  weight(term, topic) * idf(term)
## return dataframe df_idf with schema ['MLTC', topic1, ..., topicl]
def compute_terms_topics_associations(lda_model, topics_count, list_of_ltcs, bows):

    ## create dataframe with schema ['MLTC', 'topic_1',  'topic_2', ..., 'topic_k']
    topics_columns = [ "topics_"+str(i) for i in range(topics_count)]

    cols = ['MLTC']
    [ cols.append(t) for t in topics_columns]
    weighted_topics_df = pd.DataFrame(columns=cols)
    weighted_topics_df['MLTC'] = list_of_ltcs.columns

    ## get native term weights from LDA
    term_weights = lda_model.show_topics(num_words=300, formatted=False)

    ## step 1: populate weighted_topics_df with native LDA term weight
    for t in range(len(topics_columns)):
        weights_for_topic = [ x for (name, x) in term_weights[t][1]]
        weighted_topics_df[topics_columns[t]] = weights_for_topic

    ## step 2: calculate idf for each term, add idf as new column, calculate weighted terms and add them as new columns
    term_occur = term_occurrences(bows) ## term --> number of occurrences

    idf = calculate_idf(list_of_ltcs.columns, len(bows), term_occur)

    weighted_topics_df['idf'] = idf
    
    sum = 0
    for i in range(len(topics_columns)):
        sum += weighted_topics_df[topics_columns[i]]

    weighted_topics_df['sums'] = sum
    
    topics_columns = [ "weighted_topics_"+str(i) for i in range(topics_count)]

    for i in range( len(topics_columns)):
        weighted_topics_df[topics_columns[i]] = weighted_topics_df[cols[i+1]] / weighted_topics_df['sums']* idf

    weighted_topics_df['term_occurrences'] = term_occur.values()
    
    return weighted_topics_df, topics_columns





# bream down a bow into its stages, for instance:
#  bow = ['OA', 'skin_ulcer', 'dermatitis']
#  stages = [['OA'], ['OA', 'skin_ulcer'], ['OA', 'skin_ulcer', 'dermatitis']]
def getStages(bow):
        stages = []
        for i in range(len(bow)):
            stage = stages.append(bow[0:i+1])
        return stages






def pprint(trajectory):
    for bowStageId in trajectory.keys():
        print("stage {0}: {1}".format(bowStageId, trajectory[bowStageId]))
        