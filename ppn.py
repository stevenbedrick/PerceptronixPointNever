#!/usr/bin/env python -O
# Copyright (C) 2014 Kyle Gorman
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Perceptronix Point Never: a perceptron-based part-of-speech tagger

from __future__ import division

import logging
import jsonpickle

from time import time
from numpy import ones, zeros
from collections import defaultdict
from numpy.random import permutation
# timeit tests suggest that for generating a random list of indices of the
# sort used to randomize order of presentation, this is much faster than
# `random.shuffle`; if for some reason you are unable to deploy `numpy`,
# it should not be difficult to modify the code to use `random` instead.

from lazyweight import LazyWeight
from decorators import Listify, Setify
from features import *

## defaults and (pseudo)-globals
VERSION_NUMBER = 0.2
TRAINING_ITERATIONS = 10
INF = float('inf')

## set jsonpickle to do it human-readable
jsonpickle.set_encoder_options('simplejson', separators=(',', ':'), indent=' ')

## usage string
USAGE = """Perceptronix Point Never {}, by Kyle Gorman <gormanky@ohsu.edu>

    Input arguments (exactly one required):

        -i tag         train model on data in `tagged`
        -p source      read serialized model from `source`

    Output arguments (at least one required):

        -D sink        dump serialized training model to `sink`
        -E tagged      compute accuracy on data in `tagged`
        -T untagged    tag data in `untagged`
    
    Optional arguments:

        -t t           number of training iterations (default: {})
        -h             print this message and quit
        -v             increase verbosity

Options `-i` and `-E` take whitespace-delimited "token/tag" pairs as input.
Option `-T` takes whitespace-delimited tokens (no tags) as input.
""".format(VERSION_NUMBER, TRAINING_ITERATIONS)


class PPN(object):

    """
    Perceptronix Point Never: an HMM tagger with fast discriminative 
    training using the perceptron algorithm
    """

    def __init__(self, sentences=None, T=1):
        self.time = 0
        # the (outer) keys are tags;
        # are (inner) dictionaries with feature string keys and LazyWeight values
        self.weights = defaultdict(lambda: defaultdict(LazyWeight))
        logging.info('Constructed new PPN instance.')
        if sentences:
            self.train(sentences, T)

    # alternative constructor using JSON
    
    @classmethod
    def load(cls, source):
        """
        Create new PPN instance from serialized JSON from `source`
        """
        return jsonpickle.decode(source.read())

    def dump(self, sink):
        """
        Serialize object (as JSON) and print to `sink`
        """
        print >> sink, jsonpickle.encode(self)

    @staticmethod
    @Listify
    def _get_features(sentences):
        for sentence in sentences:
            (tokens, tags) = zip(*sentence)
            yield tags, POS_token_features(tokens), POS_tag_features(tags)

    @staticmethod
    def _get_tag_index(sentence_features):
        tagset = set()
        for (tags, _, _) in sentence_features:
            tagset.update(tags)
        return {i: tag for (i, tag) in enumerate(tagset)} 
        
    def train(self, sentences, T=1):
        logging.info('Extracting input features for training.')

        # "sentfeats" is a list, each element in the list represents a sentence...
        # each sentence is a triple of three lists...
        # the first list is a list of tags, the second is a list of token_feature lists,
        # and the third is a list of tag_feature lists
        sentfeats = PPN._get_features(sentences)
        # construct dictionary mapping from tag to index in trellis
        self.tag_index = PPN._get_tag_index(sentfeats)
        # begin training
        for t in xrange(T):
            tic = time()
            epoch_right = epoch_wrong = 0
            for (gtags, tokfeats, tagfeats) in permutation(sentfeats):
                # compare hypothesized tagging to gold standard
                htags = self._feature_tag_greedy(tokfeats)
                #htags = self._feature_tag(tokfeats)
                for (htag, gtag, tokf, tagf) in zip(htags, gtags, \
                                                    tokfeats, tagfeats):
                    if htag == gtag:
                        epoch_right += 1
                        continue
                    feats = tokf + tagf
                    self._update(gtag, feats, +1)
                    self._update(htag, feats, -1)
                    epoch_wrong += 1
                self.time += 1
            # check for early convergence and compute accuracy
            if epoch_wrong == 0:
                return
            acc = epoch_right / (epoch_right + epoch_wrong)
            logging.info('Epoch {:02} acc.: {:.04f}'.format(t + 1, acc) +
                         ' ({}s elapsed).'.format(int(time() - tic)))
        logging.info('Training complete.')

    def _update(self, tag, featset, sgn):
        """
        Apply update ("reward" if `sgn` == 1, "punish" if `sgn` == -1) for
        each feature in `features` for this `tag`
        """
        tag_ptr = self.weights[tag]
        for feat in featset:
            tag_ptr[feat].update(self.time, sgn)

    def tag(self, tokens):
        """
        Tag a single `sentence` (list of tokens)
        """
        return zip(tokens, 
                   self._feature_tag_greedy(POS_token_features(tokens)))
                   #self._feature_tag(POS_token_features(tokens)))

    def _feature_tag_greedy(self, tokfeats):
        """
        Tag a sentence from a list of sets of token features; note this
        returns a list of tags, not a list of (token, tag) tuples

        Deprecated: doesn't use Viterbi decoding (though it still works
        pretty well!), or even preceding tag hypotheses
        """
        
        # tokfeats is a list of token_feature lists representing the calculated features for a sentence's tokens
        # each token_feature element is a list of computed features for a given token
        tags = []
        for featset in tokfeats: # for each "token"
            best_tag = None
            best_score = -INF
            for tag in self.tag_index.itervalues(): # for each possible tag we could assign to this token...
                tag_ptr = self.weights[tag] # tag_ptr will be a hash mapping observed feature values to weights in the context of this tag
                tag_score = sum(tag_ptr[feat].get(self.time) for
                                feat in featset)
                if tag_score > best_score: # is this the best one we've seene yet?
                    best_tag = tag
                    best_score = tag_score
            tags.append(best_tag)
        return tags

    def _feature_tag(self, tokfeats):
        """
        Tag a sentence from a list of sets of token features; note this
        returns a list of tags, not a list of (token, tag) tuples
        """
        L = len(tokfeats) # len of sentence, in tokens
        Lt = len(self.tag_index) + 1 # num possible tags, plus one for the start state
        if L == 0:
            return []
        elif L == 1:
            return self._feature_tag_greedy(tokfeats)
            # FIXME this is deprecated...
        tags = []
        trellis = zeros((L, Lt), dtype=int)
        bckptrs = -ones((L, Lt), dtype=int)
        # populate trellis with sum of token feature weights
        for (t, featset) in enumerate(tokfeats):
            for (i, tag) in self.tag_index.iteritems():
                tagptr = self.weights[tag]
                trellis[t, i] += sum(tagptr[feat].get(self.time) for
                                     feat in featset)
        # special case for first tag
        featset = tag_featset()
        for (i, tag) in self.tag_index.iteritems():
            tagptr = self.weights[tag]
            trellis[0, i] += sum(tagptr[feat].get(self.time) for feat in featset)
            backptrs[0, i] = Lt # we're using Lt (num tags + 1) as a symbolic representation of the "start" state
        # special case for second tag
        for (i, tag) in self.tag_index.iteritems():
            tagptr = self.weights[tag]
            best_inbound_arc = -1
            best_inbound_score = -INF
            for (j, previous_tag) in self.tag_index.iteritems():
                featset = tag_featset(previous_tag)
                transition_weight = sum(tagptr[feat].get(self.time) for feat in featset)
                if transition_weight > best_transition_weight:
                    best_transition_index = j
                    best_transition_weight = transition_weight
            trellis[1, i] += best_transition_weight
            backptrs[1, i] = best_transition_index
        # normal case!
        for t in xrange(2, L): # for each token, starting after the two special cases we've already handled
            for (i, tag) in self.tag_index.iteritems(): # for each tag
                tagptr = self.weights[tag]
                for (j, previous_tag) in self.tag_index.iteritems():
                    best_inbound_arc_to_previous_tag = backptrs[t - 1, j]
                    featset = tag_featset(previous_tag, self.tag_index[best_inbound_arc_to_previous_tag])
                    transition_weight = sum(tagptr[feat].get(self.time) for feat in featset)
                    if transition_weight > best_transition_weight:
                        best_transition_index = j
                        best_transition_weight = transition_weight
                trellis[t, i] += best_transition_weight
                backptrs[t, i] = best_transition_index
        # figure out where to "start" backtracing from
        t = L - 1
        state_indices = zeros(L, dtype=int)
        previous_state_index = trellis[t, :].argmax()
        while t >= 0:
            state_indices[t] = previous_state_index
            state_indices[t - 1] = backptrs[t, previous_state_index]
            t -= 1
        # look up tags using backtrace indices
        return [self.tag_index[i] for i in state_indices]

    def evaluate(self, sentences):
        """
        Compute tag accuracy of the current model using a held-out list of
        `sentence`s (list of token/tag pairs)
        """
        total = 0
        correct = 0
        for sentence in sentences:
            (tokens, gtags) = zip(*sentence)
            htags = [tag for (token, tag) in self.tag(tokens)]
            for (htag, gtag) in zip(htags, gtags):
                total += 1
                correct += (htag == gtag)
        return correct / total


if __name__ == '__main__':

    from sys import argv
    from gzip import GzipFile
    from nltk import str2tuple, untag
    from getopt import getopt, GetoptError

    from decorators import Listify

    # helpers

    @Listify
    def tag_reader(filename):
        with open(filename, 'r') as source:
            for line in source:
                yield [str2tuple(wt) for wt in line.strip().split()]

    @Listify
    def untagged_reader(filename):
        with open(filename, 'r') as source:
            for line in source:
                yield line.strip().split()

    ## parse arguments
    try:
        (optlist, args) = getopt(argv[1:], 'i:p:D:E:T:t:hv')
    except GetoptError as err:
        logging.error(err)
        exit(USAGE)
    # warn users about arguments not from opts (as this is unsupported)
    for arg in args:
        logging.warning('Ignoring command-line argument "{}"'.format(arg))
    # set defaults
    test_source = None
    tagged_source = None
    untagged_source = None
    model_source = None
    model_sink = None
    training_iterations = TRAINING_ITERATIONS
    # read optlist
    for (opt, arg) in optlist:
        if opt == '-i':
            tagged_source = arg
        elif opt == '-p':
            model_source = arg
        elif opt == '-D':
            model_sink = arg
        elif opt == '-E':
            test_source = arg
        elif opt == '-T':
            untagged_source = arg
        elif opt == '-t':
            try:
                training_iterations = int(arg)
            except ValueError:
                logging.error('Cannot parse -t arg "{}".'.format(arg))
                exit(USAGE)
        elif opt == '-h':
            exit(USAGE)
        elif opt == '-v':
            logging.basicConfig(level=logging.INFO)
        else:
            logging.error('Option {} not found.'.format(opt, arg))
            exit(USAGE)

    ## check outputs
    if not any((model_sink, untagged_source, test_source)):
        logging.error('No outputs specified.')
        exit(USAGE)

    ## run inputs
    ppn = None
    if tagged_source:
        if model_source:
            logging.error('Incompatible inputs (-i and -p) specified.')
            exit(1)
        logging.info('Training model from tagged data "{}".'.format(
                                                            tagged_source))
        try:
            ppn = PPN(tag_reader(tagged_source), training_iterations)
        except IOError as err:
            logging.error(err)
            exit(1)
    elif model_source:
        logging.info('Reading model from serialized data "{}".'.format(
                                                             model_source))
        try:
            with GzipFile(model_source, 'r') as source:
                ppn = PPN.load(source)
        except IOError as err:
            logging.error(err)
            exit(1)
    else:
        logging.error('No input specified.')
        exit(USAGE)

    ## run outputs
    if test_source:
        logging.info('Evaluating on data from "{}".'.format(test_source))
        try:
            with open(test_source, 'r') as source:
                accuracy = ppn.evaluate(tag_reader(test_source))
                print 'Accuracy: {:4f}'.format(accuracy)
        except IOError as err:
            logging.error(err)
            exit(1)
    if untagged_source:
        logging.info('Tagging data from "{}".'.format(untagged_source))
        try:
            for tokens in untagged_reader(untagged_source):
                print ' '.join(tuple2str(token, tag) for
                               (token, tag) in ppn.tag(tokens))
        except IOError as err:
            logging.error(err)
            exit(1)
    if model_sink:
        logging.info('Writing serialized data to "{}.'.format(model_sink))
        try:
            with GzipFile(model_sink, 'w') as sink:
                ppn.dump(sink)
        except IOError as err:
            logging.error(err)
            exit(1)
