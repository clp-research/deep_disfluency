# Adapted from: Hidden Markov Models in Python
# Katrin Erk, March 2013
#
# This HMM addresses the problem of disfluency/end of utternace tagging.
# It estimates the probability of a tag sequence for a given word sequence as
# follows:
#
# Say words = w1....wN
# and tags = t1..tN
#
# then
# P(tags | words) is_proportional_to  product P(ti | t{i-1}) P(wi | ti)
#
# To find the best tag sequence for a given sequence of words,
# we want to find the tag sequence that has the maximum P(tags | words)

import math
from copy import deepcopy
import numpy as np
from collections import defaultdict
import cPickle
import nltk

from hmm_utils import convert_to_dot


def log(prob):
    if prob == 0.0:
        return - float("Inf")
    return math.log(prob, 2)


def convert_to_disfluency_tag(previous, tag):
    """Returns (dis)fluency tag which is dealt with in a uniform way by
    the Markov model.
    """
    if not previous:
        previous = ""
    if "<f" in tag:
        if "rpSM" in previous or "rpM" in previous:
            # TODO change back for later ones without MID!
            return "rpM"  # can be mid repair
        return "f"
    elif "<e" in tag:
        if "rpM" in previous:
            # TODO just so as not to punish it mid-repair
            return "eR"
        return "e"
    elif "<rm-" in tag:
        if "<rpEnd" in tag:
            return "rpSE"
        return "rpSM"
    elif "<rpMid" in tag:
        return "rpM"
    elif "<rpEnd" in tag:
        return "rpE"
    print "NO TAG for" + tag


def convert_to_disfluency_trp_tag(previous, tag):
    """Returns joint a list of (dis)fluency and trp tag which is dealt with
    in a uniform way by the Markov model.
    """
    trp_tag = ""
    if "<c" in tag:  # i.e. a continuation of utterance from the previous word
        trp_tag += "c_{}"
    elif "<t" in tag:  # i.e. a first word of an utterance
        trp_tag += "t_{}"
    if "c>" in tag:  # i.e. predicts a continuation next
        trp_tag += "_c"
    elif "t>" in tag:  # i.e. predicts an end of utterance for this word
        trp_tag += "_t"
    assert trp_tag != "" and trp_tag[0] != "_" and trp_tag[-1] != "}",\
        "One or both TRP tags not given" + str(trp_tag)
    if not previous:
        previous = ""
    if "<f" in tag:
        if "rpSM" in previous or "rpM" in previous:
            # TODO change back for later ones without MID!
            if "<t" not in tag and "t>" not in tag:
                return trp_tag.format("rpM")  # can be mid repair
        return trp_tag.format("f")
    elif "<e" in tag:
        if "rpM" in previous or "rpSM" in previous:  # Not to punish mid-repair
            if "<t" not in tag and "t>" not in tag:
                return trp_tag.format("eR")
        return trp_tag.format("e")
    elif "<i" in tag:
        return trp_tag.format("i")  # This should always be t_i_t
    elif "<rm-" in tag:
        if "<rpEnd" in tag:
            return trp_tag.format("rpSE")
        return trp_tag.format("rpSM")
    elif "<rpMid" in tag:
        return trp_tag.format("rpM")
    elif "<rpEnd" in tag:
        return trp_tag.format("rpE")
    print "NO TAG for" + tag


def convert_to_disfluency_trp_tag_simple(previous, tag):
    """Returns joint a list of (dis)fluency and trp tag which is dealt with in
    a uniform way by the Markov model.
    Simpler version with fewer classes.
    """
    # print previous, tag
    trp_tag = ""
    if "<c" in tag:  # i.e. a continuation of utterance from the previous word
        trp_tag += "c_{}"
    elif "<t" in tag:  # i.e. a first word of an utterance
        trp_tag += "t_{}"
    if "c>" in tag:  # i.e. predicts a continuation next
        trp_tag += "_c"
    elif "t>" in tag:  # i.e. predicts an end of utterance for this word
        trp_tag += "_t"
    assert trp_tag != "" and trp_tag[0] != "_" and trp_tag[-1] != "}",\
        "One or both TRP tags not given" + str(trp_tag)
    if not previous:
        previous = ""
    if "<f" in tag:
        return trp_tag.format("f")
    elif "<e" in tag:
        return trp_tag.format("e")
    elif "<i" in tag:
        return trp_tag.format("i")  # This should always be t_i_t
    elif "<rm-" in tag:
        return trp_tag.format("rpSE")
    print "NO TAG for" + tag


class rnn_hmm():
    """A standard hmm model which interfaces with an rnn/lstm model that
    outputs the softmax over all labels at each time step.
    """
    def __init__(self, disf_dict, rnn=None, markov_model_file=None,
                 timing_model=True, n_history=20):
        self.rnn = rnn  # rnn used to get the probability distributions
        self.tagToIndexDict = disf_dict  # dict maps from tags -> indices
        self.n_history = n_history  # how many steps back we should store
        self.observation_tags = set(self.tagToIndexDict.keys())
        self.observation_tags.add('s')  # all tag sets need a start tag

        print "loading", markov_model_file, "Markov model"
        if 'trp' in markov_model_file:
            if 'simple' in markov_model_file:
                graph = convert_to_dot(
                    "../decoder/models/disfluency_trp_simple.csv")
                self.convert_tag = convert_to_disfluency_trp_tag_simple
            else:
                graph = convert_to_dot(
                    "../decoder/models/disfluency_trp.csv")
                self.convert_tag = convert_to_disfluency_trp_tag
        else:
            self.observation_tags.add('se')  # add end tag when in pre-seg mode
            graph = convert_to_dot("../decoder/models/disfluency_trp.csv")
            self.convert_tag = convert_to_disfluency_tag
        if timing_model:
            print "loading timing model"
            with open('../decoder/LogReg_balanced_timing_classifier.pkl',
                      'rb') as fid:
                self.timing_model = cPickle.load(fid)
            with open('../decoder/LogReg_balanced_timing_scaler.pkl',
                      'rb') as fid:
                self.timing_model_scaler = cPickle.load(fid)
            # self.simple_trp_idx2label = {0 : "<cc>",
            #                        1 : "<ct>",
            #                        2 : "<tc>",
            #                       3 : "<tt>"}
            # Only use the Inbetween and Start tags
            self.simple_trp_idx2label = {0: "<c", 1: "<t"}
        tags = []
        self.tag_set = []
        for line in graph.split("\n"):
            spl = line.replace(";", "").split()
            if not len(spl) == 3:
                continue
            assert spl[1] == "->"
            tags.append((spl[0], spl[2]))
            self.tag_set.extend([spl[0], spl[2]])
        self.tag_set = set(self.tag_set)
        print self.tag_set
        print self.observation_tags
        # print "If we have just seen 'DET', \
        # the probability of 'N' is", cpd_tags["DET"].prob("N")
        cfd_tags = nltk.ConditionalFreqDist(tags)
        self.cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)

        self.viterbi_init()  # initialize viterbi
        # print "Test: If we have just seen 'rpSM',\
        # the probability of 'f' is", self.cpd_tags["c_rpSM_c"].prob("c_f_c")

    def viterbi_init(self):
        self.best_tagsequence = []  # presume this is for a new sequence
        self.viterbi = []
        self.backpointer = []
        self.converted = []
        self.history = []

    def add_to_history(self, viterbi, backpointer, converted):
        """We store a history of n_history steps back in case we need to
        rollback.
        """
        if len(self.history) == self.n_history:
            self.history.pop(-1)
        self.history = [{"viterbi": deepcopy(viterbi),
                         "backpointer": deepcopy(backpointer),
                         "converted": deepcopy(converted)}] + self.history

    def rollback(self, n):
        """Rolling back to n back in the history."""
        # print "rollback",n
        # print len(self.history)
        self.history = self.history[n:]
        self.viterbi = self.viterbi[:len(self.viterbi)-n]
        self.backpointer = self.backpointer[:len(self.backpointer)-n]
        self.converted = self.converted[:len(self.converted)-n]
        self.best_tagsequence = self.best_tagsequence[
            :len(self.best_tagsequence)-n]

    def viterbi_step(self, softmax, word_index, sequence_initial=False,
                     timing_data=None):
        """The principal viterbi calculation for an extension to the
        input prefix, i.e. not reseting.
        """
        if sequence_initial:
            # first time requires initialization with the start of sequence tag
            first_viterbi = {}
            first_backpointer = {}
            first_converted = {}
            for tag in self.observation_tags:
                # don't record anything for the START tag
                # print tag
                if tag == "s" or tag == 'se':
                    continue
                # print word_index
                # print softmax.shape
                # print self.tagToIndexDict[tag]
                # print softmax[word_index][self.tagToIndexDict[tag]]
                first_viterbi[tag] = log(self.cpd_tags["s"].
                                         prob(self.convert_tag("s", tag))) + \
                    log(softmax[word_index][self.tagToIndexDict[tag]])
                # no timing bias to start
                first_backpointer[tag] = "s"
                first_converted[tag] = self.convert_tag("s", tag)
                assert first_converted[tag] in self.tag_set,\
                    first_converted[tag]
            # store first_viterbi (the dictionary for the first word)
            # in the viterbi list, and record that the best previous tag
            # for any first tag is "s" (start of sequence tag)
            self.viterbi.append(first_viterbi)
            self.backpointer.append(first_backpointer)
            self.converted.append(first_converted)
            self.add_to_history(first_viterbi, first_backpointer,
                                first_converted)
            return
        # else we're beyond the first word
        # start a new dictionary where we can store, for each tag, the prob
        # of the best tag sequence ending in that tag
        # for the current word in the sentence
        this_viterbi = {}
        # we also store the best previous converted tag
        this_converted = {}  # added for the best converted tags
        # start a new dictionary we we can store, for each tag,
        # the best previous tag
        this_backpointer = {}
        # prev_viterbi is a dictionary that stores, for each tag, the prob
        # of the best tag sequence ending in that tag
        # for the previous word in the sentence.
        # So it stores, for each tag, the probability of a tag sequence
        # up to the previous word
        # ending in that tag.
        prev_viterbi = self.viterbi[-1]
        prev_converted = self.converted[-1]
        # for each tag, determine what the best previous-tag is,
        # and what the probability is of the best tag sequence ending.
        # store this information in the dictionary this_viterbi
        if timing_data and self.timing_model:
            # X = self.timing_model_scaler.transform(np.asarray(
            # [timing_data[word_index-2:word_index+1]]))
            # TODO may already be an array
            # print "calculating timing"
            X = self.timing_model_scaler.transform(np.asarray([timing_data]))
            softmax_timing = self.timing_model.predict_proba(X)
        for tag in self.observation_tags:
            # don't record anything for the START/END tag
            if tag in ["s", "se"]:
                continue
            # joint probability calculation:
            # if this tag is X and the current word is w, then
            # find the previous tag Y such that
            # the best tag sequence that ends in X
            # actually ends in Y X
            # that is, the Y that maximizes
            # prev_viterbi[ Y ] * P(X | Y) * P( w | X)
            # The following command has the same notation
            # that you saw in the sorted() command.
            best_previous = None
            best_prob = log(0.0)  # has to be -inf for log numbers
            # the inner loop which makes this quadratic complexity
            # in the size of the tag set
            for prevtag in prev_viterbi.keys():
                # the best converted tag, needs to access the previous one
                prev_converted_tag = prev_converted[prevtag]
                # TODO there could be several conversions for this tag
                converted_tag = self.convert_tag(prev_converted_tag, tag)
                assert converted_tag in self.tag_set, tag + " " + \
                    converted_tag + " prev:" + str(prev_converted_tag)
                tag_prob = self.cpd_tags[prev_converted_tag].prob(
                    converted_tag)
                if tag_prob > 0:
                    # TODO for now treating this like a {0,1} constraint
                    tag_prob = log(1.0)
                    test = converted_tag.lower()
                    if "rpe" in test:  # boost for end tags
                        tag_prob = tag_prob
                    elif "rpsm_" in test:
                        tag_prob = tag_prob
                    if timing_data and self.timing_model:
                        for k, v in self.simple_trp_idx2label.items():
                            if v in tag:
                                timing_tag = k
                                found = True
                                break
                        if not found:
                            raw_input("warning")
                        # using the prob from the timing classifier
                        # array over the different classes
                        timing_prob = softmax_timing[0][timing_tag]
                        tag_prob = log(0.5 * timing_prob)
                else:
                    tag_prob = log(0.0)
                prob = prev_viterbi[prevtag] + tag_prob + \
                    log(softmax[word_index][self.tagToIndexDict[tag]])
                if prob >= best_prob:
                    best_converted = converted_tag
                    best_previous = prevtag
                    best_prob = prob
            this_converted[tag] = best_converted
            this_viterbi[tag] = best_prob
            # the most likely preceding tag for this current tag
            this_backpointer[tag] = best_previous
        # done with all tags in this iteration
        # so store the current viterbi step
        self.viterbi.append(this_viterbi)
        self.backpointer.append(this_backpointer)
        self.converted.append(this_converted)
        self.add_to_history(this_viterbi, this_backpointer, this_converted)
        return

    def get_best_tag_sequence(self):
        """Returns the best tag sequence from the input so far.
        """
        inc_prev_viterbi = deepcopy(self.viterbi[-1])
        inc_best_previous = max(inc_prev_viterbi.keys(),
                                key=lambda prevtag: inc_prev_viterbi[prevtag])
        assert(inc_prev_viterbi[inc_best_previous]) != log(0),\
            "highest likelihood is 0!"
        inc_best_tag_sequence = [inc_best_previous]
        # invert the list of backpointers
        inc_backpointer = deepcopy(self.backpointer)
        inc_backpointer.reverse()
        # go backwards through the list of backpointers
        # (or in this case forward, we have inverted the backpointer list)
        inc_current_best_tag = inc_best_previous
        for bp in inc_backpointer:
            inc_best_tag_sequence.append(bp[inc_current_best_tag])
            inc_current_best_tag = bp[inc_current_best_tag]

        inc_best_tag_sequence.reverse()
        return inc_best_tag_sequence

    def viterbi(self, softmax, incremental_best=False):
        """Standard non incremental (sequence-level) viterbi over softmax input

        Keyword arguments:
        softmax -- the emmision probabilities of each step in the sequence,
        array of width n_classes
        incremental_best -- whether the tag sequence prefix is stored for
        each step in the sequence (slightly 'hack-remental'
        """
        incrementalBest = []
        sentlen = len(softmax)
        self.viterbi_init()

        for word_index in range(0, sentlen):
            self.viterbi_step(softmax, word_index, word_index == 0)
            # INCREMENTAL RESULTS (hack-remental. doing it post-hoc)
            # the best result we have so far, not given the next one
            if incremental_best:
                inc_best_tag_sequence = self.get_best_tag_sequence()
                incrementalBest.append(deepcopy(inc_best_tag_sequence[1:]))
        # done with all words/input in the sentence/sentence
        # find the probability of each tag having "se" next (end of utterance)
        # and use that to find the overall best sequence
        prev_converted = self.converted[-1]
        prev_viterbi = self.viterbi[-1]
        best_previous = max(prev_viterbi.keys(),
                            key=lambda prevtag: prev_viterbi[prevtag] +
                            log(self.cpd_tags[prev_converted[prevtag]].
                            prob("se")))
        self.best_tagsequence = ["se", best_previous]
        # invert the list of backpointers
        self.backpointer.reverse()
        # go backwards through the list of backpointers
        # (or in this case forward, we've inverted the backpointer list)
        # in each case:
        # the following best tag is the one listed under
        # the backpointer for the current best tag
        current_best_tag = best_previous
        for bp in self.backpointer:
            self.best_tagsequence.append(bp[current_best_tag])
            current_best_tag = bp[current_best_tag]
        self.best_tagsequence.reverse()
        if incremental_best:
            # NB also consumes the end of utterance token! Last two the same
            incrementalBest.append(self.best_tagsequence[1:-1])
            return incrementalBest
        return self.best_tagsequence[1:-1]

    def viterbi_incremental(self, soft_max, a_range=None,
                            changed_suffix_only=False, timing_data=None):
        """Given a new softmax input, output the latest labels.
        Effectively incrementing/editing self.best_tagsequence.

        Keyword arguments:
        changed_suffix_only -- boolean, output the changed suffix of
        the previous output sequence of labels.
            i.e. if before this function is called the sequence is
            [1:A, 2:B, 3:C]
            and after it is
            [1:A, 2:B, 3:E, 4:D]
            then output is:
            [3:E, 4:D]
            (TODO maintaining the index/time spans is important
            to acheive this, even if only externally)
        """
        previous_best = deepcopy(self.best_tagsequence)
        # print "previous best", previous_best
        if not a_range:
            # if not specified consume the whole soft_max input
            a_range = (0, len(soft_max))
        for i in xrange(a_range[0], a_range[1]):
            self.viterbi_step(soft_max, i, sequence_initial=self.viterbi == [],
                              timing_data=timing_data)
            # slice the input if multiple steps
            # get the best tag sequence we have so far
        self.best_tagsequence = self.get_best_tag_sequence()
        # print "best_tag", self.best_tagsequence
        if changed_suffix_only:
            # print "current best", self.best_tagsequence
            # only output the suffix of predictions which has changed-
            # TODO needs IDs to work
            for r in range(1, len(self.best_tagsequence)):
                if r > len(previous_best)-1 or \
                        previous_best[r] != self.best_tagsequence[r]:
                    return self.best_tagsequence[r:]
        return self.best_tagsequence[1:]

if __name__ == '__main__':
    def load_tags(filepath):
        """Returns a tag dictionary from word to a n int indicating index
        by an integer.
        """
        tag_dictionary = defaultdict(int)
        f = open(filepath)
        for line in f:
            l = line.strip('\n').split(",")
            tag_dictionary[l[1]] = int(l[0])
        f.close()
        return tag_dictionary
    tags = load_tags("../data/tag_representations/swbd1_trp_tags.csv")
    intereg_ind = len(tags.keys())
    tags["<i/><cc>"] = intereg_ind  # add the interregnum tag
    print tags
    h = rnn_hmm(tags, markov_model_file="models/disfluency_trp.csv")
    h.viterbi_init()
    s0 = [0] * (len(tags.keys())-1)
    s0[tags["<e/><tc>"]] = 0.5
    s1 = [0] * (len(tags.keys())-1)
    s1[tags["<e/><cc>"]] = 0.5
    s1[tags['<rm-2/><rpEndSub/><ct>']] = 0.5
    softmax2 = np.asarray([s0, s1], dtype="float32")
    print softmax2
    print softmax2.shape
    softmax = np.concatenate((softmax2,
                              softmax2[:, tags["<e/><cc>"]].
                              reshape(softmax2.shape[0], 1)), 1)
    print softmax
    print softmax.shape
    x2 = h.viterbi_incremental(softmax, [0, 2])
    s = 0
    for d in h.viterbi:
        print "stage", s
        s += 1
        for key, value in d.items():
            print key, value
    print x2
