# Importing modules
import os

import pandas as pd
import re
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
import pyLDAvis.gensim
import pickle
import pyLDAvis

num_topics = 10 # Build LDA model

nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['robot'])

feedback = pd.read_csv('./data/data.csv')# Print head
print(feedback.columns)
print(feedback.head())

feedback['feedback_text_processed'] = feedback['text'].map(lambda x: re.sub('[,\.!?]', '', x))# Convert the titles to lowercase
feedback['feedback_text_processed'] = feedback['feedback_text_processed'].map(lambda x: x.lower())# Print out the first rows of papers

print(feedback['feedback_text_processed'].head())

def word_cloud():
    # Join the different processed titles together.
    long_string = ','.join(list(feedback['feedback_text_processed'].values))# Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')# Generate a word cloud
    wordcloud.generate(long_string)# Visualize the word cloud
    wordcloud.to_file("wordcloud.png")


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def topic_analysis_preprocessing(texts):
    data = texts.feedback_text_processed.values.tolist()
    data_words = list(sent_to_words(data)) # remove stop words
    data_words = remove_stopwords(data_words)
    print(data_words[:1][0][:30])

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)  # Create Corpus
    texts = data_words  # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]  # View
    print(corpus[:1][0][:30])
    return corpus, id2word



def topic_analysis_training(corpus, id2word):

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    return lda_model, doc_lda

corpus, id2word = topic_analysis_preprocessing(feedback)
lda_model, doc_lda = topic_analysis_training(corpus, id2word)

# Visualize the topics

LDAvis_data_filepath = os.path.join('./data/ldavis_prepared_'+str(num_topics))

# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, LDAvis_data_filepath +'.html')

word_cloud()