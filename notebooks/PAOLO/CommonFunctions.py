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






### Create a word cloud with calculated association strength (idf * relative weights)

def createWordClouds(df_idf):
    # Wordcloud of Top N LTCs for each topic based on association strength

    i = 1
    for j in df_idf.columns[6:10]:

        ltc_dic = pd.Series(df_idf[j].values,index=df_idf.MLTC).to_dict()

        colrs = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        cloud = WordCloud(background_color='white',
                        width=2500,
                        height=1800,
                        max_words=10,
                        colormap='tab10',
                        color_func=lambda *args, **kwargs: colrs[i],
                        prefer_horizontal=1.0)

        topics = df_idf[[j]] # Since we have fixed the topics to 4, we can change this to make it dynamic

        if i < 9:
            # fig.add_subplot(ax)
            topic_words = ltc_dic
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            titl = 'Topic ' + str(j[-1])
            plt.gca().set_title(titl, fontdict=dict(size=16))
            plt.gca().axis('off')


        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        plt.show()
        i = i+1

        # fig.savefig("../images/topic0_relative_weight.png", dpi=60)

        
# Topics generation

# in: bow is the list of bag of words
# in: topics_count is the number of topics to be generated
# returns lda-model

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
    with open(LDA_MODEL, 'wb') as f:
        pickle.dump(lda_model, f)
        
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
        