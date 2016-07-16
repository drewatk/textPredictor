from nltk.corpus import reuters
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, LaplaceProbDist
from nltk import word_tokenize
import time

class ngram():
    def __init__(self):
        self._corpus = reuters.words()
        self._train()

    def _train(self):
        print 'Training models...'
        start_time = time.time()
        self._bigram_cfd = ConditionalFreqDist()
        self._trigram_cfd = ConditionalFreqDist()
        self._quadgram_cfd = ConditionalFreqDist()
        prev_word = None
        prev_2_word = None
        prev_3_word = None
        for word in self._corpus:
            if word.isalpha():
                self._bigram_cfd[prev_word][word] += 1
                self._trigram_cfd[tuple([prev_2_word, prev_word])][word] += 1
                #print tuple([prev_3_word, prev_2_word, prev_word])
                self._quadgram_cfd[tuple([prev_3_word, prev_2_word, prev_word])][word] += 1
                prev_3_word = prev_2_word
                prev_2_word = prev_word
                prev_word = word

        self._bigram_cpd = ConditionalProbDist(self._bigram_cfd, LaplaceProbDist, len(self._bigram_cfd.conditions()))
        self._trigram_cpd = ConditionalProbDist(self._trigram_cfd, LaplaceProbDist, len(self._trigram_cfd.conditions()))
        self._quadgram_cpd = ConditionalProbDist(self._quadgram_cfd, LaplaceProbDist, len(self._quadgram_cfd.conditions()))

        print 'Models trained, took %s seconds' % (time.time() - start_time)

    def next_word(self, context):
        context = word_tokenize(context)
        word = self._quadgram_cpd[tuple(context[-3:])].max()
        return word

n = ngram()
sentence = 'I have a'
for x in range (100):
    sentence += (' ' + n.next_word(sentence))

print sentence
