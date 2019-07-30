import re
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import wordpunct_tokenize 


filenames = ['conversations_1.csv', 'conversations_2.csv']

text_data = pd.DataFrame()

for file in filenames:
    text_data = text_data.append(pd.read_csv(os.path.join('E:/Akshay/text', file),encoding='latin1'))


text_data = text_data.groupby('conversation-id').agg({'content':', '.join}).reset_index()
    
text_data['content'] = text_data['content'].fillna('No data available')
text_data['content'] = text_data['content'].map(lambda x: str(x).lower())

def remove_sp_chr(val):
    return ' '.join(re.findall('\w+', val))

text_data['content'] = text_data['content'].map(lambda x: remove_sp_chr(x))

def remove_stopwords(doc):
    word_tokens = word_tokenize(doc) 
    filtered_sentence = ' '.join([w for w in word_tokens if w not in stop_words])
    return filtered_sentence

stop_words = stopwords.words("english")
text_data['content'] = text_data['content'].map(lambda x: remove_stopwords(x))

text_data = text_data[text_data['content']!='info']


#### filter out english words

words = set(nltk.corpus.words.words())
def english_filter(sent):
    return ' '.join(w for w in wordpunct_tokenize(sent) \
             if w.lower() in words or w.isalpha())

text_data['content'] = text_data['content'].map(lambda x: english_filter(x))

# Frequency distribution of words in content column
freq_dist = text_data.content.str.split(expand=True).stack().value_counts()

def filter_gt3(val):
    return ' '.join([w for w in val.split(' ') if len(w) >2])

text_data['content'] = text_data['content'].map(lambda x: filter_gt3(x))

## imporvosing stopwords list
text_data = text_data[~text_data.content.str.contains('menu')]
text_data = text_data[~text_data['content'].isin(['','yes', 'hello', 'thanks'])]

#text_data = text_data[~text_data.content.str.contains('yes')]


# Lemmatizing text
from textblob import Word
text_data['content'] = text_data['content'].map(lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))
#text_data.head()



## TF-IDF vectorization 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000, # keep top 1000 terms 
max_df = 0.5, 
smooth_idf=True)

X = vectorizer.fit_transform(text_data)

X.shape # check shape of the document-term matrix
    



## LDA Topic modelling

from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)

svd_model.fit(X)

len(svd_model.components_)


## 

terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Topic "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")



##### Wordclouds 

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

text = pd.Series(['I', 'I', 'I', 'I', 'I', 'Love', 'Love','Love', 'Love', 'You', 'You', 'Piu', 'Piu', 'Piu'])
#text = text_data#.content.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black').generate(str(text))

fig = plt.figure(
    figsize = (5, 5),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

