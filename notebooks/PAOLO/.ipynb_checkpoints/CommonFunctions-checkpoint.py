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


# bream down a bow into its stages, for instance:
#  bow = ['OA', 'skin_ulcer', 'dermatitis']
#  stages = [['OA'], ['OA', 'skin_ulcer'], ['OA', 'skin_ulcer', 'dermatitis']]
def getStages(bow):
        stages = []
        for i in range(len(bow)):
            stage = stages.append(bow[0:i+1])
        return stages


# look up each term in bowStage. add up all partial association strength for each topic, generating an array of size K = number of topic
# def assoc(bowStage, rwidf):
def assoc(bowstage, rwidf):
    assocVector = np.zeros(topics_count)
    for term in bowstage:
        row = rwidf.loc[terms_topics_df['MLTC'] == term]
#         print(row[['rwidf1', 'rwidf2', 'rwidf3', 'rwidf4']])
        assocVector[0] += row['rwidf1']
        assocVector[1] += row['rwidf2']
        assocVector[2] += row['rwidf3']
        assocVector[3] += row['rwidf4']
    return assocVector


##########################
### compute the tensor that holds the patient-topic association __for incremental bag of terms in the patient's history__
##########################

## main method to compute the 'tensor' as a nested dictionary:
# bows = list(bow)
# bow = list(bowStage)
# bowStage = list(term)
# term --> association vector of size K = number of topics
# so:
#    all_patients_traj = { id(bow): one_patient_trajectory}
#    one_patient_trajectory = { id(bowStage): assoc vector}
#  to use hashing we need to create an id for each bow (id=patient) and one id for each stage in that patient's history.
#  note that these are not the native patient IDs which are lost at this point
#
# return all_patients_traj
#
def computeTrajectoryAssociations(bows, rwidf):

    bowID2bow = dict()  ## need to use bowIDs as hash keys so this dict maps bowID to the actual bow content
    bowId = 1 # makes hashing possible
    all_patients_traj = dict()  ## top level dict
    
    for bow in bows:    # for each bag of word (each patient)
#         print("processing bowId {0}: [{1}]".format(bowId, bow))
        
        bowID2bow[bowId] = bow
        traj = all_patients_traj[bowId] = dict()  ## individual trajectory is itself a dict()
        bowStageId = 1
        for bowStage in getStages(bow):  # compute association vector for each of its stages
#             print("processing bowStage [{0}]".format(bowStage))
            
            traj[bowStageId]  = assoc(bowStage, rwidf)
#             print("vector for bowStageId [{0}]: {1}".format(bowStageId, traj[bowStageId]))
            bowStageId += 1
#         print("trajectory: {0}\n".format(traj))
        bowId += 1
        if bowId % 1000 == 0:
            print("{0} patients processed".format(bowId))
            
    ## save main trajectories data structure

    with open(ALL_TRAJECTORIES, 'wb') as f:
        pickle.dump(trajectories, f)

    with open(BOWID2BOW, 'wb') as f:
        pickle.dump(bowId2bow, f)

    return all_patients_traj, bowID2bow


def pprint(trajectory):
    for bowStageId in trajectory.keys():
        print("stage {0}: {1}".format(bowStageId, trajectory[bowStageId]))
        
        