from nltk.corpus import wordnet

syns = wordnet.synsets("program")

#just the word
print(syns[0].lemmas()[0].name())
# synset
print(syns[0].name())


#defination
print(syns[0].definition())

#exaples
print(syns[0].examples())


synonyms=[]
antonyms=[]

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


#semantic_simialrity


w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")


print(w1.wup_similarity(w2))













