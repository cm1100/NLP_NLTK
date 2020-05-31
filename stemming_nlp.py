# i was taking ride in car
#i was riding in the car

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

e_words = ["python","pythoner","pythoning","pythoned","pythonly"]

for w in e_words:
    print(ps.stem(w))

new_text = "it is very important to pythonly while you are pythoning with python . All pythoners have pythoned atleast once"

words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))