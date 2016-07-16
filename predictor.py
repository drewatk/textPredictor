from nltk.corpus import udhr 
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, MLEProbDist 
from nltk.tokenize import word_tokenize 
from random import choice

cfd = ConditionalFreqDist()

paragraph = 'He is first credited with ending a revolt of religious dissenters in the vicinity of Caesarea. He was then supposedly able to do so without resorting to force. He is then credited as the strategos (general) responsible for capturing an enemy fortress, which was considered impregnable. Who were the enemy barbarians is left unclear, though they could have been hostile Arabs. He is thirdly mentioned leading a force of about 20 men in opening a pass which was held closed by Arab attacks. Again supposedly without a battle. He is next mentioning restoring Byzantine control over Iotabe (Tiran Island), previously occupied by neighbouring tribesmen. Aratius was finally able to locate the stronghold of these tribesmen on the mainland, attack it and capture it. He is particularly praised for the revenue brought in by Iotabe, through the customs paid there'

words = word_tokenize(paragraph) 

prev_word = None
for word in udhr.words():
    if word.isalpha():
        cfd[prev_word][word] += 1
        prev_word = word

word = 'therefore'

#cfd.plot()
#print cfd.unicode_repr()
#print cfd.N()

cpd = ConditionalProbDist(cfd, MLEProbDist, len(cfd.conditions()))

word = "United"
suggestion = []
for i in range(30):
    suggestion.append(word)
    word = cpd[word].max()

print " ".join(suggestion)
