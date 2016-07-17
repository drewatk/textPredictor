#! /usr/bin/python

from nltk.corpus import reuters
from nltk.probability import FreqDist, ConditionalFreqDist, ConditionalProbDist, LaplaceProbDist
from nltk import word_tokenize
import time
import cPickle as pickle
from os.path import isfile

class ngram(object):
    def __init__(self, load_from_disk=True):
        self._corpus = reuters.words()

        self._unigram_fd = FreqDist()
        self._bigram_cfd = ConditionalFreqDist()
        self._trigram_cfd = ConditionalFreqDist()
        self._quadgram_cfd = ConditionalFreqDist()

        self._unigram_pd = None
        self._bigram_cpd = None
        self._trigram_cpd = None
        self._quadgram_cpd = None

        if load_from_disk:
            self._load_models()
        else:
            self._train()

    def _train(self):
        print 'Training models...'
        start_time = time.time()

        prev_word = None
        prev_2_word = None
        prev_3_word = None
        for word in self._corpus:
            if word.isalpha():
                self._unigram_fd[word] += 1
                self._bigram_cfd[prev_word][word] += 1
                self._trigram_cfd[tuple([prev_2_word, prev_word])][word] += 1
                self._quadgram_cfd[tuple([prev_3_word, prev_2_word, prev_word])][word] += 1
                prev_3_word = prev_2_word
                prev_2_word = prev_word
                prev_word = word

        self._unigram_pd = LaplaceProbDist(self._unigram_fd, bins=self._unigram_fd.N())
        self._bigram_cpd = ConditionalProbDist(self._bigram_cfd, LaplaceProbDist, bins=len(self._bigram_cfd.conditions()))
        self._trigram_cpd = ConditionalProbDist(self._trigram_cfd, LaplaceProbDist, bins=len(self._trigram_cfd.conditions()))
        self._quadgram_cpd = ConditionalProbDist(self._quadgram_cfd, LaplaceProbDist, bins=len(self._quadgram_cfd.conditions()))
        
        print 'Models trained, took %s seconds' % (time.time() - start_time)
        
        self._save_models()

    def _save_models(self):
        print 'Saving Models to disk...'
        start_time = time.time()

        pickle.dump(self._unigram_pd, open('./unigram_pd.p', 'w'))
        pickle.dump(self._bigram_cpd , open('./bigram_cpd.p', 'w'))
        pickle.dump(self._trigram_cpd, open('./trigram_cpd.p', 'w'))
        pickle.dump(self._quadgram_cpd, open('./quadgram_cpd.p', 'w'))

        print 'Models saved, took %s seconds' % (time.time() - start_time)

    def _load_models(self):
        
        if not (isfile('./unigram_pd.p') and isfile('./bigram_cpd.p') and isfile('./trigram_cpd.p') and isfile('./quadgram_cpd.p')):
            self._train()
            return

        print 'Loading Models from disk...'
        start_time = time.time()

        self._unigram_pd = pickle.load(open('./unigram_pd.p', 'r'))
        self._bigram_cpd = pickle.load(open('./bigram_cpd.p', 'r'))
        self._trigram_cpd = pickle.load(open('./trigram_cpd.p', 'r'))
        self._quadgram_cpd = pickle.load(open('./quadgram_cpd.p', 'r'))

        print 'Models loaded, took %s seconds' % (time.time() - start_time)

    def next_word(self, context):
        context = word_tokenize(context)
        word = self._quadgram_cpd[tuple(context[-3:])].max()
        return word

def main():
    n = ngram()
    sentence = 'the dollar was'
    for x in range (100):
        sentence += (' ' + n.next_word(sentence))

    print sentence
    
if __name__ == "__main__":
    main()