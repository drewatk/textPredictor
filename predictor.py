from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, LaplaceProbDist

cfd = ConditionalFreqDist()

prev_word = None
for word in brown.words():
    if word.isalpha():
        cfd[prev_word][word] += 1
        prev_word = word

cpd = ConditionalProbDist(cfd, LaplaceProbDist, len(cfd.conditions()))

word = 'fortress'

suggestion = []
for i in range(30):
    suggestion.append(word)
    word = cpd[word].generate()

print " ".join(suggestion)
