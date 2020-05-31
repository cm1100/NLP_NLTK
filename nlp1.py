import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# tokeniziing : word tokenizers...  sentence tokenizers
#lexicon and corporas
#corpora : body of text ex:medical journals , presidential speeches, English language
#lexicon - words and their means
#investor-speak.....  regualr english speak

#investor speak 'bull' = someone who is positive about the market
#english speak bull : an animal

example_text = "MR. cm ,Hello there i am there ? The weather is great and i am good . Lets play it "
print(sent_tokenize(example_text))
print(word_tokenize(example_text))