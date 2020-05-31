import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

t_text = state_union.raw("2005-GWBush.txt")
#print(t_text)
s_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(t_text)
print(custom_sent_tokenizer)

tokenized = custom_sent_tokenizer.tokenize(s_text)
#print(tokenized)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            named = nltk.ne_chunk(tagged,binary=True)

            named.draw()

    except Exception as e:
        print(str(e))

process_content()


'''
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian

'''