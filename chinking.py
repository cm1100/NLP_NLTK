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

            # CHUNKING : making group meanigful data

            chunKGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|>{ """

            chunKParser = nltk.RegexpParser(chunKGram)
            #print(chunKParser)
            chunked = chunKParser.parse(tagged)

            print(chunked)


    except Exception as e:
        print(str(e))

process_content()