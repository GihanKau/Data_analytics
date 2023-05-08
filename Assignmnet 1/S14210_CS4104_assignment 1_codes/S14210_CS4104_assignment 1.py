"""
Name        : B.M.G.G.K. Rajapaksha
Faculty     : Faculty of Science, Bioinformatics
Index NO.   : S14210
Date        : 25/07/2022

Task        : Assignment 1; An algorithm to determine the document frequency (tfidf classifier)
Input       : 1. A set of .txt documents (8) related to 3 different news topics
              2. Topics (3) of the documents

Output      : 1. Cosine Similarity matrix
              2. Categorization of the documents according to their similarity and topics

"""

# STEP 1 -Import libraries

from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# STEP 2 - Open documents

with open('doc 1.txt', 'r') as doc_1:
    doc_1 = doc_1.read()

with open('doc 2.txt', 'r') as doc_2:
    doc_2 = doc_2.read()

with open('doc 3.txt', 'r') as doc_3:
    doc_3 = doc_3.read()

with open('doc 4.txt', 'r') as doc_4:
    doc_4 = doc_4.read()

with open('doc 5.txt', 'r') as doc_5:
    doc_5 = doc_5.read()

with open('doc 6.txt', 'r') as doc_6:
    doc_6 = doc_6.read()

with open('doc 7.txt', 'r') as doc_7:
    doc_7 = doc_7.read()

with open('doc 8.txt', 'r') as doc_8:
    doc_8 = doc_8.read()

# News topics

topic_1 = 'Hurricane Gilbert Heads Toward Dominican Coast'
topic_2 = 'IRA terrorist attack'
topic_3 = 'McDonald\'s Opens First Restaurant in China'

# STEP 3 - take the corpus into a list
corpus = [doc_1, doc_2, doc_3, doc_4, doc_5, doc_6, doc_7, doc_8, topic_1, topic_2, topic_3]

# STEP 4 - Fit the corpus in to the vectorizer with preprocessing parameters

# convert all texts into lowercase and remove stop words and initiate the TfidfVectorizer
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')

# fit the corpus into the vectorizer
tf_idf_matrix = vectorizer.fit_transform(corpus)

# take the vocabulary into a sorted dictionary
sorted_dict = OrderedDict(sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1], reverse=False))
print('Sorted vocabulary\n', sorted_dict, '\n')
print('Length of the vocabulary - ', len(vectorizer.vocabulary_), '\n')

# STEP 5 - Cosine similarity matrix

cosine_sim = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

# transform the similarity matrix into a dataframe
df = pd.DataFrame(cosine_sim, columns=['doc_1', 'doc_2', 'doc_3', 'doc_4', 'doc_5', 'doc_6', 'doc_7', 'doc_8',
                                       'topic_1', 'topic_2', 'topic_3'],
                  index=['doc_1', 'doc_2', 'doc_3', 'doc_4', 'doc_5', 'doc_6', 'doc_7', 'doc_8',
                         'topic_1', 'topic_2', 'topic_3'])

pd.set_option('display.max_columns', 11)
print('Cosine similarity matrix\n', df, '\n')

# Identify documents related to the each topic
print('Identify documents related to each topic')
row = 0
for row in range(8):
    if df.iloc[row:row + 1, 8:11].max(axis=1)[0] == 0:
        check_most_sim_doc = df.iloc[row:row + 1, 0:8]
        most_sim_doc = check_most_sim_doc.T.apply(lambda x: x.nlargest(2).idxmin())[0]

        while df.iloc[df.index == most_sim_doc, 8:11].max(axis=1)[0] <= 0:
            check_most_sim_doc = df.iloc[df.index == most_sim_doc, 0:8]
            most_sim_doc = check_most_sim_doc.T.apply(lambda x: x.nlargest(2).idxmin())[0]

        print(df.columns[row], '-', df.iloc[df.index == most_sim_doc, 8:11].idxmax(axis=1)[0])

    else:
        print(df.columns[row], '-', df.iloc[row:row + 1, 8:11].idxmax(axis=1)[0])

# END