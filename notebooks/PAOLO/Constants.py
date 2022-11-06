DATA_PATH= '/Users/npm65/OneDrive - Newcastle University/NIHR-AI-MLTC-M/DATA/ukb45840-main dataset/ukbb45840_lists_and_topics/'
WD = '/Users/npm65/OneDrive - Newcastle University/NIHR-AI-MLTC-M/CODE/cluster-assignments/MLTC-Patient_modelling/'

LTC_BINARY = DATA_PATH+'ltc_matrix_binary_mm4.tsv'

## generated cached data
TERMS_REL_WEIGHTS_IDF = WD+'data/terms_rel_weights_idf.csv'
BOWs = WD+'data/generated/BOWs.pkl'
LTCs = WD+'data/generated/LTCSs.pkl'

## these are specific for each number of clusters
LDA_MODEL = WD+'data/generated/lda_model.pkl'
ALL_TRAJECTORIES = WD+'data/generated/trajectories.pkl'
BOWID2BOW  = WD+'data/generated/BowId2bow.pkl'
