# coding=utf-8
from __future__ import unicode_literals, print_function

import pickle
import random
import re
import string
from collections import defaultdict, Counter


class Raver:
    CONTEXT_USAGE = 1

    def __init__(self, filenames):
        if filenames is None:
            return
        if type(filenames) == str:
            filenames = [filenames]
        first_in_post = []
        last_in_post = []
        self.first_in_sentence = []
        last_in_sentence = []
        bigrams = []
        trigrams = []
        word_count = defaultdict(int)
        punctuation = '.!?'#re.sub(':', '', string.punctuation)
        i = 0
        for filename, use_as_first in filenames:
            with open(filename, 'r') as texts:
                for i, line in enumerate(texts.readlines()):
                    if i % 1000 == 0:
                        print('\r%d' % i, end='')
                    new_sentence = True
                    words = re.findall(u'([0-9a-zA-Zа-яА-Яё@\.,!\?:_/\-\']+)', line.decode('utf8'))
                    words = map(lambda x: x.lower(), words)
                    while len(words) > 0 and words[0].rstrip(punctuation) == '':
                        words = words[1:]
                    while len(words) > 0 and words[-1].rstrip(punctuation) == '':
                        words = words[:-1]

                    if len(words) < 1:
                        continue

                    if use_as_first:
                        first_in_post.append(words[0].rstrip(punctuation))
                    last_in_post.append(words[-1].rstrip(punctuation))
                    prevprev = None
                    prev = None
                    for word in words:

                        if new_sentence:
                            if use_as_first:
                                self.first_in_sentence.append(word.rstrip(punctuation))
                            new_sentence = False
                        if word[-1] in '.!?':
                            new_sentence = True
                        word = word.rstrip(punctuation)
                        word_count[word] += 1
                        if prev is not None:
                            bigrams.append((prev, word))
                            if prevprev is not None:
                                trigrams.append((prevprev, prev, word))

                        prevprev, prev = prev, word
                        if new_sentence:
                            last_in_sentence.append(word)
                            prevprev, prev = None, None
        print('\r%d' % i)
        self.word_count = dict(word_count)
        #
        self.last_in_sentence_stats = {k: (float(v) / self.word_count[k]) for k, v in Counter(last_in_sentence).items()}
#        self.last_in_post_stats = {k: (float(v) / word_count[k]) for k, v in Counter(last_in_post).items()}
        self.last_in_post_stats = {}
        for k, v in Counter(last_in_post).items():
            try:
                wc = self.word_count[k]
                self.last_in_post_stats[k] = float(v) / wc
            except:
                # print(k, v, self.word_count[k])
                raise
        trigram_stats = defaultdict(list)
        for f, s, t in trigrams:
            if self.word_count[f] > 1 and self.word_count[s] > 1 and self.word_count[t] > 1:
                trigram_stats[(f, s)].append(t)
        self.trigram_stats = dict(trigram_stats)

        bigram_stats = defaultdict(list)
        for f, s in bigrams:
            if (f, s) in trigram_stats:
                bigram_stats[f].append(s)
        self.bigram_stats = dict(bigram_stats)

        print('WC', len(self.word_count), 'BI', len(self.bigram_stats), 'TRI', len(self.trigram_stats))

    def _gen_word(self, first, second, context):
        try:
            return self._context_choice(self.trigram_stats[(first, second)], context)
        except:
            try:
                return self._context_choice(self.bigram_stats[second], context)
            except:
                return None

    def _context_choice(self, array, context, mult = 1):
        if context is not None:
            random.shuffle(context)
            for word in context:
                if word in array and random.uniform(0, 1) < Raver.CONTEXT_USAGE * mult:
                    # print('Gotcha', word)
                    return word

        return random.choice(array)

    def _gen_first_second(self, context):
        first, second = None, None
        for _ in range(10):
            try:
                first = self._context_choice(self.first_in_sentence, context, .5)
                second = self._context_choice(self.bigram_stats[first], context)
                break
            except:
                pass
        if first is None or second is None:
            raise Exception('((((')

        return first, second

    def gen_sentence(self, context=None, as_string=True):
        if context is None:
            context = []
        f, s = self._gen_first_second(context)
        sentence = [f, s]
        for x in range(150):
            t = self._gen_word(f, s, context)
            if t is None:
                break
            sentence.append(t)
            try:
                if random.uniform(0, 1.) < self.last_in_sentence_stats[t]:
                    break
            except:
                pass
            f, s = s, t

        if as_string:
            return ' '.join(sentence)
        else:
            return sentence

    def gen_post(self):
        post = []
        end = False
        while not end:
            sentence = self.gen_sentence(False)
            try:
                if random.uniform(0, 1.) > self.last_in_post_stats[sentence[-1]]:
                    end = True
            except:
                pass
            sentence[-1] += '.'
            post += sentence
        return ' '.join(post)

    def save(self, fname='rave'):
        pickle.dump(self, open(fname, 'w'))


def load(fname):
    return pickle.load(open(fname, 'r'))
