# MLTC-patient trajectory
In this study, 
* we explored techniques to optimise the interpretability of clusters of LTCs by applying existing topic modelling techniques to patients with MLTC from the UK Biobank, to discover meaningful connections between long-term conditions.
* Aim to quantify the association of each individual with each of the topics generated by LDA.

* Using association strength based on the patients medical history, we observe the patients trajectory between the clusters

, potentially discovering risk indicators that we can use to optimise the prediction of a patient’s trajectory. We will then visualise the final clusters to analyse the complex combinations of co-occurring long-term conditions.


### Folder Structure
MLTC
    Data - Contains the datasets used and produced in the research.
    Images - Contains the images used in the research.
    notebooks - Contains the code used in the research.
    requirements.txt - Python package dependencies for this research. 


### To run the the Analysis

* First install all the dependencies of python using pip. (Given in requirements.txt)
* Structure the folders as per the folder structure mentioned above.
* Run the note books.
    Note
    * BLOCK 3 OF 'LDA_Association_Str.ipynb' does not run in one go. Patient data needs to be divided into multiple parts for easy processing. - Instruction written in the code block.

    * Block 3 produces a dataset with all the details of the patient. It is already stored with the name - 'Results.csv'



### Notebooks description

1: LDA_Association_Str.ipynb
Contains the code used to produced LDA topic modeling and for creating association metric.

2: trajectory_modelling.ipynb
Contains the code to visualize and process the patients trajectory.


### Dataset Generated

ltc_events_all_patients_ukbb45840.tsv - UK Biobank data.
ltc_matrix_binary_mm4.tsv - Transformed UK Biobank binary data.

Results.csv - Contain patient details of their association strength with each of the clusters at the ned of their disease history.
final.csv - Contains normal dominance of each disease within the LDA clusters along with relative weight of each diseases as decribed in the study.
final_x_idf.csv - Contains the association strength calculated using our association metric for each disease.
corr_table.csv - Contains correlation data between 203 LTCs.
lda_tuning_results.csv - Contains the result form hyperparameter tuning.