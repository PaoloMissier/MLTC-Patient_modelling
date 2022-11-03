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
        
    return list_of_patients



# Build a dataframe with each term's relative weight within each topic
# in: lda_model

def compute_terms_topics_associations(lda_model):
    # Creating a matrix of disease/MLTC/word distribution within each cluster

    colnames_raw = ['MLTC', 'topic 1', 'topic 2', 'topic 3', 'topic 4']
    colnames_final = ['rwidf1', 'rwidf2', 'rwidf3', 'rwidf4']

    ## new df with one column per topic
    df_word_weightage_bytopic = pd.DataFrame(columns=colnames_raw)

    ## for each topic t
    for t in range(len(colnames_raw)-1):
        ## new df with schema [MLTC, topic_t]
        
#         print("t ={0}".format(t))

        df_cal = pd.DataFrame(columns=['MLTC', colnames_raw[t+1]])  
        
        nm = list()
        dst = list()
        ## for each condition
        ## 
        for i in range(203):
            ## https://radimrehurek.com/gensim/models/ldamodel.html?highlight=show_topics#gensim.models.ldamodel.LdaModel.show_topics
            ## return list of {str, tuple of (str, float)}
            name, dist = lda_model.show_topics(num_words=203, formatted=False)[t][1][i]
            nm.append(name)   ## name of condition
            dst.append(dist)  ## scores of term for each topic
        df_cal[colnames_raw[t+1]] = dst
        df_cal['MLTC'] = nm
        df_cal.sort_values(by=['MLTC'], ignore_index=True, inplace=True)

        df_word_weightage_bytopic[colnames_raw[t+1]] = df_cal[colnames_raw[t+1]]
    df_word_weightage_bytopic['MLTC'] = df_cal['MLTC']

#     print(df_word_weightage_bytopic.head)
    
    df_idf = df_word_weightage_bytopic.copy()

    # Calculating disease term frequency in our corpus
    disease_count = []

    ## for each disease:
    for i in df_idf['MLTC']:
        count = 0
        for k in bows:   # for each patient bow k
            if i in k:   # found occurrence of term i in k
                count = count +1            
        disease_count.append(count)

    # disease_count is a list that now becomes a new column 'idf'
    df_idf['occurrences'] = disease_count

    # Calculating inverse document frequency
    df_idf['idf'] = 0. ## init new column
    
    for i in range(len(df_idf['occurrences'])):        
        df_idf['idf'].loc[i] = math.log10(len(bows) / (df_idf['occurrences'].loc[i] + 1))
    df_idf[colnames_final[0]] = ''
    df_idf[colnames_final[1]] = ''
    df_idf[colnames_final[2]] = ''
    df_idf[colnames_final[3]] = ''

    # Calculating relative weight for each word/disease

    for i, rows in df_idf.iterrows():

        sum = rows[1]+rows[2]+rows[3]+rows[4]
        df_idf[colnames_final[0]][i] = rows[1]/(sum)
        df_idf[colnames_final[1]][i] = rows[2]/(sum)
        df_idf[colnames_final[2]][i] = rows[3]/(sum)
        df_idf[colnames_final[3]][i] = rows[4]/(sum)

#     ## add word counts
#     df = []

#     for i in df_idf['MLTC']:
#         count = 0
#         for k in bows:
#             if i in k:
#                 count = count +1            
#         df.append(count)

#     df_idf['word occurence in docs'] = df

    # Add IDF to the terms relative weights
    df_idf[colnames_final[0]] = df_idf[colnames_final[0]] * df_idf['occurrences']
    df_idf[colnames_final[1]] = df_idf[colnames_final[1]] * df_idf['occurrences']
    df_idf[colnames_final[2]] = df_idf[colnames_final[2]] * df_idf['occurrences']
    df_idf[colnames_final[3]] = df_idf[colnames_final[3]] * df_idf['occurrences']

    df_idf.sort_values(by=['occurrences'], ascending=False)
    
    ## cache for faster future use
    df_idf.to_csv(TERMS_REL_WEIGHTS_IDF)    

    return df_idf


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


        