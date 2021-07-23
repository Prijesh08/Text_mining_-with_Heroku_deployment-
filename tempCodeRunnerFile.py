import spacy
en_model=spacy.load('en_core_web_sm')
sw_spacy=en_model.Defaults.stop_words
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(min_df=2 ,stop_words=sw_spacy , ngram_range=(1,2))
tfidf_transformer = TfidfTransformer()

s = "it is good product"

X_new_counts = count_vect.transform(s)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)