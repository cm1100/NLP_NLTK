import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
sampletext = gutenberg.raw("bible-kjv.txt")

st = sent_tokenize(sampletext)

