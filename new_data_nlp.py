import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.tokenize import word_tokenize
import io

short_pos = io.open("datasets/positive.txt",encoding="latin-1").read()
short_neg = io.open("datasets/negative.txt",encoding="latin-1").read()

documents =[]

for r in short_pos.split('\n'):
    documents.append((r,"pos"))

for r in short_neg.split('\n'):
    documents.append((r,"neg"))

all_words=[]


short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())


all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

p_n = open("new_data.pi","wb")
pickle.dump(featuresets,p_n)
p_n.close()


# set that we'll train our classifier with

training_set = featuresets[:10000]

# set that we'll test against.
testing_set = featuresets[10000:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)