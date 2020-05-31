import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from nltk.corpus import movie_reviews
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self,classifiers):
        self.classifiers=classifiers

    def classify(self,featureset):
        votes = []
        for c in self.classifiers:
            v = c.classify(featureset)

            votes.append(v)
        #print(votes)
        return mode(votes)


    def confidence(self,featureset):
        votes = []
        for c in self.classifiers:
            v = c.classify(featureset)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf







read_c = open("new_data.pi","rb")
featureset = pickle.load(read_c)
read_c.close()

training_set = featureset[:10000]
test_set = featureset[10000:]

MNB = SklearnClassifier(MultinomialNB())
MNB.train(training_set)
print(nltk.classify.accuracy(MNB,test_set))


#GNB = SklearnClassifier(GaussianNB())
#GNB.train(training_set)
#print(nltk.classify.accuracy(GNB,test_set))


BNB = SklearnClassifier(BernoulliNB())
BNB.train(training_set)
print(nltk.classify.accuracy(BNB,test_set))


LR = SklearnClassifier(LogisticRegression(max_iter=300))
LR.train(training_set)
print(nltk.classify.accuracy(LR,test_set))


SGDC = SklearnClassifier(SGDClassifier())
SGDC.train(training_set)
print(nltk.classify.accuracy(SGDC,test_set))


svc= SklearnClassifier(SVC())
svc.train(training_set)
print(nltk.classify.accuracy(svc,test_set))


Lsvc = SklearnClassifier(LinearSVC())
Lsvc.train(training_set)
print(nltk.classify.accuracy(Lsvc,test_set))


#Nsvc = SklearnClassifier(NuSVC)
#Nsvc.train(training_set)
#print(nltk.classify.accuracy(Nsvc,test_set))

save_all = open("all_algo.al","wb")
pickle.dump(MNB,save_all)
pickle.dump(BNB,save_all)
pickle.dump(LR,save_all)
pickle.dump(SGDC,save_all)
pickle.dump(svc,save_all)
pickle.dump(Lsvc,save_all)
#pickle.dump(Nsvc)
save_all.close()

''''

read_all = open("all_algo.al","rb")
MBN = pickle.load(read_all)

BNB = pickle.load(read_all)
LR = pickle.load(read_all)
SGDC = pickle.load(read_all)
svc = pickle.load(read_all)
Lsvc = pickle.load(read_all)
read_all.close()

read_n = open("naive_base.pickle","rb")
nbp= pickle.load(read_n)
read_n.close()



print(nltk.classify.accuracy(nbp,test_set))
print(nltk.classify.accuracy(MBN,test_set))
print(nltk.classify.accuracy(BNB,test_set))
print(nltk.classify.accuracy(LR,test_set))
print(nltk.classify.accuracy(SGDC,test_set))
print(nltk.classify.accuracy(svc,test_set))
print(nltk.classify.accuracy(Lsvc,test_set))'''

voted_classifier = VoteClassifier([MNB,BNB,LR,SGDC,svc,Lsvc])
print("new  ",nltk.classify.accuracy(voted_classifier,test_set))

print("Classification",voted_classifier.classify(test_set[0][0]),"confidence:" , voted_classifier.confidence(test_set[0][0]))

