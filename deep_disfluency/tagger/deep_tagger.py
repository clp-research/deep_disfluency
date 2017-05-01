from __future__ import division
import argparse
import numpy as np
import cPickle
import os
from copy import deepcopy
import time
from nltk.tag import CRFTagger

from deep_disfluency.language_model.ngram_language_model \
    import KneserNeySmoothingModel
from deep_disfluency.utils.tools \
    import convert_from_inc_disfluency_tags_to_eval_tags
from deep_disfluency.utils.tools import verify_dialogue_data_matrices
from deep_disfluency.load.load import load_tags
from deep_disfluency.rnn.elman import Elman
from deep_disfluency.rnn.lstm import LSTM
from deep_disfluency.rnn.test_if_using_gpu import test_if_using_GPU
from deep_disfluency.decoder.hmm import FirstOrderHMM
from deep_disfluency.decoder.noisy_channel import SourceModel


def process_arguments(config=None,
                      exp_id=None,
                      heldout_file="../data/disfluency_detection/" +
                      "switchboard/swbd_heldout_partial_data.csv",
                      test_file="../data/disfluency_detection/" +
                      "switchboard/swbd_test_partial_data.csv",
                      use_saved=None,
                      hmm=None,
                      verbose=True):
    """Loads arguments for an experiment from a config file

    Keyword arguments:
    config -- the config file location, default None
    exp_id -- the experiment ID name, default None
    """
    parser = argparse.ArgumentParser(description='This script trains a RNN for\
        disfluency detection and saves the best models and results to disk.')
    parser.add_argument('-c', '--config', type=str,
                        help='The location of the config file.',
                        default=config)
    parser.add_argument('-e', '--exp_id', type=str,
                        help='The experiment number from which to load \
                            arguments from the config file.',
                        default=exp_id)
    parser.add_argument('-v', '--heldout_file', type=str,
                        help='The path to the validation file.',
                        default=heldout_file)
    parser.add_argument('-t', '--test_file', type=str,
                        help='The path to the test file.',
                        default=test_file)
    parser.add_argument('-m', '--use_saved_model', type=int,
                        help='Epoch number of the pre-trained model to load.',
                        default=use_saved)
    parser.add_argument('-hmm', '--decoder_file', type=str,
                        help='Path to the hmm file.',
                        default=hmm)
    parser.add_argument('-verb', '--verbose', type=bool,
                        help='Whether to output training progress.',
                        default=verbose)
    args = parser.parse_args()

    header = [
        'exp_id',  # experiment id
        'model_type',  # can be elman/lstm/mt_elman/mt_lstm
        'lr',  # learning rate
        'decay',  # decay on the learning rate if improvement stops
        'seed',  # random seed
        'window',  # number of words in the context window (backwards only)
        'bs',  # number of backprop through time steps
        'emb_dimension',  # dimension of word embedding
        'n_hidden',  # number of hidden units
        'n_epochs',  # maximum number of epochs
        'train_data',  # which training data
        'partial_words',  # whether partial words are in the data
        'loss_function',  # default will be nll, unlikely to change
        'reg',  # regularization type
        'pos',  # whether pos tags or not
        'n_acoustic_features',  # number of acoustic features used per word
        'n_language_model_features',  # number of language model features used
        'embeddings',  # embedding files, if any
        'update_embeddings',  # whether the embeddings should be updates
        'batch_size',  # batch size, 'word' or 'utterance'
        'word_rep',  # the word to index mapping filename
        'pos_rep',  # the pos tag to index mapping filename
        'tags',  # the output tag representations used
        'decoder_type',  # which type of decoder
        'utts_presegmented',  # whether utterances are pre-segmented
        'do_utt_segmentation'  # whether we do combined end of utt detection
        ]
    # print header
    if args.config:
        for line in open(args.config):
            # print line
            features = line.strip("\n").split(",")
            if features[0] != str(args.exp_id):
                continue
            for i in range(1, len(header)):
                feat_value = features[i].strip()  # if string
                if feat_value == 'None':
                    feat_value = None
                elif feat_value == 'True':
                    feat_value = True
                elif feat_value == 'False':
                    feat_value = False
                elif header[i] in ['lr']:
                    feat_value = float(feat_value)
                elif header[i] in ['seed', 'window', 'bs',
                                   'emb_dimension', 'n_hidden', 'n_epochs',
                                   'n_acoustic_features',
                                   'n_language_model_features']:
                    feat_value = int(feat_value)
                setattr(args, header[i], feat_value)
    return args


def get_last_n_features(feature, current_words, idx, n=3):
    """For the purposes of timing info, get the timing, word or pos
    values  of the last n words (default = 3).
    """
    if feature == "words":
        position = 0
    elif feature == "POS":
        position = 1
    elif feature == "timings":
        position = 2
    else:
        raise Exception("Unknown feature {}".format(feature))
    start = max(0, (idx - n) + 1)
    # print start, idx + 1
    return [triple[position] for triple in
            current_words[start: idx + 1]]


class IncrementalTagger(object):
    """A generic incremental tagging object which can deal with incremental
    word-by-word (and revokable) input. It doesn't do anything.
    """

    def __init__(self, config_file, config_number, saved_model_folder=None):
        print "initializing Tagger"
        # initialize (word, pos) tuple graph
        self.window_size = 0
        self.word_graph = []
        self.output_tags = []
        self.timing_model = None
        self.timing_model_scaler = None

    def load_model_from_folder(self, saved_model_folder, model_type):
        """Assumes some machine learning model needs to loaded.
        """
        raise NotImplementedError

    def tag_new_word(self, word, pos=None, timing=None):
        """Extends current context.
        """
        raise NotImplementedError

    def rollback(self, backwards):
        """Revoke the right frontier of the input and labels back backwards.
        """
        self.word_graph = self.word_graph[:-backwards]
        self.output_tags = self.output_tags[:-backwards]

    def tag_new_prefix(self, prefix, rollback=0):
        self.rollback(rollback)
        for w, p, t in prefix:
            self.tag_new_word(w, p, t)

    def reset(self):
        self.word_graph = [(self.word_to_index_map["<s>"],
                            self.pos_to_index_map["<s>"], 0)] \
                            * (self.window_size - 1)
        self.output_tags = []


class DeepDisfluencyTagger(IncrementalTagger):
    """A deep-learning driven incremental disfluency tagger
    (and optionally utterance-segmenter).
    """
    def __init__(self, config_file=None,
                 config_number=None,
                 saved_model_dir=None,
                 pos_tagger=None,
                 language_model=None,
                 pos_language_model=None,
                 edit_language_model=None,
                 timer=None,
                 timer_scaler=None):

        if not config_file:
            config_file = os.path.dirname(os.path.realpath(__file__)) +\
                "/../experiments/experiment_configs.csv"
            config_number = 35
            print "No config file, using default", config_file, config_number

        super(DeepDisfluencyTagger, self).__init__(config_file,
                                                   config_number,
                                                   saved_model_dir)
        print "Processing args from config file..."
        self.args = process_arguments(config_file,
                                      config_number,
                                      use_saved=False,
                                      hmm=True)
        print "Intializing model from args..."
        self.model = self.init_model_from_config(self.args)

        # load a model from a folder if specified
        if saved_model_dir:
            print "loading saved weights from", saved_model_dir
            self.load_model_params_from_folder(saved_model_dir,
                                               self.args.model_type)
        else:
            print "WARNING no saved model params, needs training..."

        if pos_tagger:
            print "loading POS tagger..."
            self.pos_tagger = pos_tagger
        elif self.args.pos:
            print "No POS tagger specified,loading default CRF switchboard one"
            self.pos_tagger = CRFTagger()
            tagger_path = os.path.dirname(os.path.realpath(__file__)) +\
                "/../feature_extraction/crfpostagger"
            self.pos_tagger.set_model_file(tagger_path)

        if self.args.n_language_model_features > 0 or \
                'noisy_channel' in self.args.decoder_type:
            print "training language model..."
            self.init_language_models(language_model,
                                      pos_language_model,
                                      edit_language_model)

        if timer:
            print "loading timer..."
            self.timing_model = timer
            self.timing_model_scaler = timer_scaler
        else:
            # self.timing_model = None
            # self.timing_model_scaler = None
            print "No timer specified, using default switchboard one"
            timer_path = os.path.dirname(os.path.realpath(__file__)) +\
                '/../decoder/timing_models/' + \
                'LogReg_balanced_timing_classifier.pkl'
            with open(timer_path, 'rb') as fid:
                self.timing_model = cPickle.load(fid)
            timer_scaler_path = os.path.dirname(os.path.realpath(__file__)) +\
                '/../decoder/timing_models/' + \
                'LogReg_balanced_timing_scaler.pkl'
            with open(timer_scaler_path, 'rb') as fid:
                self.timing_model_scaler = cPickle.load(fid)
                # TODO a hack
                # self.timing_model_scaler.scale_ = \
                #    self.timing_model_scaler.std_.copy()

        print "loading decoder..."
        hmm_dict = deepcopy(self.tag_to_index_map)
        # add the interegnum tag
        if "disf" in self.args.tags:
            intereg_ind = len(hmm_dict.keys())
            interreg_tag = \
                "<i/><cc/>" if "uttseg" in self.args.tags else "<i/>"
            hmm_dict[interreg_tag] = intereg_ind  # add the interregnum tag

        # decoder_file = os.path.dirname(os.path.realpath(__file__)) + \
        #     "/../decoder/model/{}_tags".format(self.args.tags)
        noisy_channel = None
        if 'noisy_channel' in self.args.decoder_type:
            noisy_channel = SourceModel(self.lm, self.pos_lm)
        self.decoder = FirstOrderHMM(
            hmm_dict,
            markov_model_file=self.args.tags,
            timing_model=self.timing_model,
            timing_model_scaler=self.timing_model_scaler,
            constraint_only=True,
            noisy_channel=noisy_channel
            )

        # getting the states in the right shape
        self.state_history = []
        self.softmax_history = []
        # self.convert_to_output_tags = get_conversion_method(self.args.tags)
        self.reset()

    def init_language_models(self, language_model=None,
                             pos_language_model=None,
                             edit_language_model=None):
        clean_model_dir = os.path.dirname(os.path.realpath(__file__)) +\
            "/../data/lm_corpora"
        if language_model:
            self.lm = language_model
        else:
            print "No language model specified, using default switchboard one"
            lm_corpus_file = open(clean_model_dir +
                                  "/swbd_disf_train_1_clean.text")
            lines = [line.strip("\n").split(",")[1] for line in lm_corpus_file
                     if "POS," not in line and not line.strip("\n") == ""]
            split = int(0.9 * len(lines))
            lm_corpus = "\n".join(lines[:split])
            heldout_lm_corpus = "\n".join(lines[split:])
            lm_corpus_file.close()
            self.lm = KneserNeySmoothingModel(
                                        order=3,
                                        discount=0.7,
                                        partial_words=self.args.partial_words,
                                        train_corpus=lm_corpus,
                                        heldout_corpus=heldout_lm_corpus,
                                        second_corpus=None)
        if pos_language_model:
            self.pos_lm = pos_language_model
        elif self.args.pos:
            print "No pos language model specified, \
            using default switchboard one"
            lm_corpus_file = open(clean_model_dir +
                                  "/swbd_disf_train_1_clean.text")
            lines = [line.strip("\n").split(",")[1] for line in lm_corpus_file
                     if "POS," in line and not line.strip("\n") == ""]
            split = int(0.9 * len(lines))
            lm_corpus = "\n".join(lines[:split])
            heldout_lm_corpus = "\n".join(lines[split:])
            lm_corpus_file.close()
            self.pos_lm = KneserNeySmoothingModel(
                                        order=3,
                                        discount=0.7,
                                        partial_words=self.args.partial_words,
                                        train_corpus=lm_corpus,
                                        heldout_corpus=heldout_lm_corpus,
                                        second_corpus=None)
        if edit_language_model:
            self.edit_lm = edit_language_model
        else:
            edit_lm_corpus_file = open(clean_model_dir +
                                       "/swbd_disf_train_1_edit.text")
            edit_lines = [line.strip("\n").split(",")[1]
                          for line in edit_lm_corpus_file
                          if "POS," not in line and not line.strip("\n") == ""]
            edit_split = int(0.9 * len(edit_lines))
            edit_lm_corpus = "\n".join(edit_lines[:edit_split])
            heldout_edit_lm_corpus = "\n".join(edit_lines[edit_split:])
            edit_lm_corpus_file.close()
            self.edit_lm = KneserNeySmoothingModel(
                                        train_corpus=edit_lm_corpus,
                                        heldout_corpus=heldout_edit_lm_corpus,
                                        order=2,
                                        discount=0.7)

    def init_model_from_config(self, args):
        # for feat, val in args._get_kwargs():
        #     print feat, val, type(val)
        if not test_if_using_GPU():
            print "Warning: not using GPU, might be a bit slow"
            print "\tAdjust Theano config file ($HOME/.theanorc)"
        print "loading tag to index maps..."
        label_path = os.path.dirname(os.path.realpath(__file__)) +\
            "/../data/tag_representations/{}_tags.csv".format(args.tags)
        word_path = os.path.dirname(os.path.realpath(__file__)) +\
            "/../data/tag_representations/{}.csv".format(args.word_rep)
        pos_path = os.path.dirname(os.path.realpath(__file__)) +\
            "/../data/tag_representations/{}.csv".format(args.pos_rep)
        self.tag_to_index_map = load_tags(label_path)
        self.word_to_index_map = load_tags(word_path)
        self.pos_to_index_map = load_tags(pos_path)
        self.model_type = args.model_type
        vocab_size = len(self.word_to_index_map.keys())
        emb_dimension = args.emb_dimension
        n_hidden = args.n_hidden
        n_extra = args.n_language_model_features + args.n_acoustic_features
        n_classes = len(self.tag_to_index_map.keys())
        self.window_size = args.window
        n_pos = len(self.pos_to_index_map.keys())
        update_embeddings = args.update_embeddings
        lr = args.lr
        print "Initializing model of type", self.model_type, "..."
        if self.model_type == 'elman':
            model = Elman(ne=vocab_size,
                          de=emb_dimension,
                          nh=n_hidden,
                          na=n_extra,
                          n_out=n_classes,
                          cs=self.window_size,
                          npos=n_pos,
                          update_embeddings=update_embeddings)
            self.initial_h0_state = model.h0.get_value()
            self.initial_c0_state = None

        elif self.model_type == 'lstm':
            model = LSTM(ne=vocab_size,
                         de=emb_dimension,
                         n_lstm=n_hidden,
                         na=n_extra,
                         n_out=n_classes,
                         cs=self.window_size,
                         npos=n_pos,
                         lr=lr,
                         single_output=True,
                         cost_function='nll')
            self.initial_h0_state = model.h0.get_value()
            self.initial_c0_state = model.c0.get_value()
        else:
            raise NotImplementedError('No model init for {0}'.format(
                self.model_type))
        return model

    def load_model_params_from_folder(self, model_folder, model_type):
        if model_type in ["lstm", "elman"]:
            self.model.load_weights_from_folder(model_folder)
            self.initial_h0_state = self.model.h0.get_value()
            if model_type == "lstm":
                self.initial_c0_state = self.model.c0.get_value()
        else:
            raise NotImplementedError('No weight loading for {0}'.format(
                model_type))

    def tag_new_word(self, word, pos=None, timing=None,
                     proper_name_pos_tags=["NNP", "NNPS", "CD", "LS",
                                           "SYM", "FW"]):
        # 0. Add new word to word graph
        word = word.lower()
        if not pos and self.pos_tagger:
            pos = self.pos_tagger.tag([])  # TODO
        if pos:
            pos = pos.upper()
            if pos in proper_name_pos_tags and "$unc$" not in word:
                word = "$unc$" + word
            if pos not in self.pos_to_index_map.keys():
                # print "unknown pos", pos
                pos = "<unk>"
        if word not in self.word_to_index_map.keys():
            # print "unknown word", word
            word = "<unk>"
        # print "New word:", word, pos
        self.word_graph.append((self.word_to_index_map[word],
                                self.pos_to_index_map[pos],
                                timing
                                ))
        # print "word graph ****"
        # for w, p, t in self.word_graph:
        #     print "**", w, p, t
        # print "****"
        # 1. load the saved internal rnn state
        if self.state_history == []:
            c0_state = self.initial_c0_state
            h0_state = self.initial_h0_state
        else:
            if self.model_type == "lstm":
                c0_state = self.state_history[-1][0][-1]
                h0_state = self.state_history[-1][1][-1]
            elif self.model_type == "elman":
                h0_state = self.state_history[-1][-1]

        if self.model_type == "lstm":
            self.model.load_weights(c0=c0_state,
                                    h0=h0_state)
        elif self.model_type == "elman":
            self.model.load_weights(h0=h0_state)
        else:
            raise NotImplementedError("no history loading for\
                             {0} model".format(self.model_type))

        # 2. do the softmax output
        word_window = get_last_n_features("words", self.word_graph,
                                          len(self.word_graph)-1,
                                          n=self.window_size)
        pos_window = get_last_n_features("POS", self.word_graph,
                                         len(self.word_graph)-1,
                                         n=self.window_size)
        # print "word_window, pos_window", word_window, pos_window
        if self.model_type == "lstm":
            h_t, c_t, s_t = self.model.\
                soft_max_return_hidden_layer([word_window], [pos_window])
            self.softmax_history.append(s_t)
            if len(self.state_history) == 20:  # just saving history
                self.state_history.pop(0)  # pop first one
            self.state_history.append((c_t, h_t))
        elif self.model_type == "elman":
            h_t, s_t = self.model.soft_max_return_hidden_layer([word_window],
                                                               [pos_window])
            self.softmax_history.append(s_t)
            if len(self.state_history) == 20:
                self.state_history.pop(0)  # pop first one
            self.state_history.append(h_t)
        else:
            raise NotImplementedError("no softmax implemented for\
                                 {0} model".format(self.model_type))

        softmax = np.concatenate(self.softmax_history)
        # 3. do the decoding on the softmax
        if "disf" in self.args.tags:
            edit_tag = "<e/><cc>" if "uttseg" in self.args.tags else "<e/>"
            # print self.tag_to_index_map[edit_tag]
            adjustsoftmax = np.concatenate((
                    softmax,
                    softmax[
                        :,
                        self.tag_to_index_map[edit_tag]].reshape(
                            softmax.shape[0],
                            1)), 1)
        else:
            adjustsoftmax = softmax
        last_n_timings = get_last_n_features("timings", self.word_graph,
                                             len(self.word_graph)-1,
                                             n=3)
        new_tags = self.decoder.viterbi_incremental(
            adjustsoftmax, a_range=(len(adjustsoftmax)-1,
                                    len(adjustsoftmax)),
            changed_suffix_only=True,
            timing_data=last_n_timings,
            words=[word])
        # print "new tags", new_tags
        self.output_tags = self.output_tags[:len(self.output_tags) -
                                            (len(new_tags) -
                                             1)] + new_tags

        # 4. convert to general output format
        if "simple" in self.args.tags:
            for p in range(
                    len(self.output_tags) -
                    (len(new_tags) + 1),
                    len(self.output_tags)):
                rps = self.output_tags[p]
                self.output_tags[p] = rps.replace(
                    'rm-0',
                    'rps id="{}"'.format(p))
                if "<i" in self.output_tags[p]:
                    self.output_tags[p] = self.output_tags[p].\
                        replace("<e/>", "").replace("<i", "<e/><i")
        else:
            # new_words = [word]
            words = get_last_n_features("words", self.word_graph,
                                        len(self.word_graph)-1,
                                        n=len(self.word_graph) -
                                        (self.window_size-1))
            self.output_tags = convert_from_inc_disfluency_tags_to_eval_tags(
                        self.output_tags,
                        words,
                        start=len(self.output_tags) -
                        (len(new_tags)),
                        representation=self.args.tags)
        return self.output_tags

    def tag_utterance(self, utterance):
        if not self.args.utts_presegmented:
            raise NotImplementedError("Tagger trained on unsegmented data,\
            please call tag_prefix(words) instead.")
        # non segmenting
        self.reset()  # always starts in initial state
        if not self.args.pos:  # no pos tag model
            utterance = [(w, None, t) for w, p, t in utterance]
            print "Warning: not using pos tags as not pos tag model"
        if not self.args.timings:
            utterance = [(w, p, None) for w, p, t in utterance]
            print "Warning: not using timing durations as no timing model"
        for w, p, t in utterance:
            if self.args.pos:
                self.tag_word(w, pos=p, timing=t)
        return self.output_tags

    def rollback(self, backwards):
        IncrementalTagger.rollback(self, backwards)
        self.state_history = self.state_history[:-backwards]
        self.softmax_history = self.softmax_history[:-backwards]
        self.decoder.rollback(backwards)

    def reset(self):
        IncrementalTagger.reset(self)
        self.state_history = []
        self.softmax_history = []
        self.decoder.viterbi_init()
        if self.model_type == "lstm":
            self.model.load_weights(c0=self.initial_c0_state,
                                    h0=self.initial_h0_state)
        elif self.model_type == "elman":
            self.model.load_weights(h0=self.initial_h0_state)

    def train_net(self, dialogue_matrices, model_dir):
        if not verify_dialogue_data_matrices(
                                        dialogue_matrices,
                                        self.args.word_to_index_map,
                                        self.args.pos_to_index_map,
                                        self.args.tags_to_index_map,
                                        self.args.n_language_model_features,
                                        self.args.n_acoustic_features):
            raise Exception("Dialogue vectors in wrong format! See README.md.")
        start = 1  # by default start from the first epoch
        for e in range(start, self.args.n_epochs + 1):
            tic = time.time()
            epoch_folder = model_dir + "/epoch_{}".format(e)
            if not os.path.exists(epoch_folder):
                os.mkdir(epoch_folder)
            
#             
#             if s['use_saved_model']:
#                 print "loading stored model and weights- not actually training from " + epochfolder
#                 self.model.load_weights_from_folder(epochfolder)
#                 #if s['model'] == "lstm":
#                 #    rnn.load_weights_from_folder(epochfolder)
#                 #else:
#                 #    emb, Wx, Wh, W, bh, b, h0  = rnn.load(epochfolder)
#                 #    rnn.load_weights(emb, Wx, Wh, W, bh, b, h0)
#             else:
#                 if s['pos']:
#                     train_loss = self.model.fit(data['train'],
#                                                 s['clr'],
#                                                 acoustic=args.acoustic,
#                                                 load_data=True)
#                 else:
#                     pass
#             
#             if s['verbose']: # output final learning time
#                 print '[learning] epoch %i >>'%(e),'completed in %.2f (sec) <<\r'%(time.time()-tic),

    def train_decoder(self, tag_file):
        return NotImplementedError

    def save_net_weights_to_dir(self, dir_path):
        return NotImplementedError

    def save_decoder_model(self, dir_path):
        return NotImplementedError
