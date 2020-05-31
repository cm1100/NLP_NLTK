from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

e_snetence = "this is an example showing filteration"
stop_words = set(stopwords.words("english"))

print(stop_words)

words = word_tokenize(e_snetence)

filtered = []

for w in words:
    if w not in stop_words:
        filtered.append(w)

filtered_sentence = [w for w in words if not w in stop_words]

print(filtered)