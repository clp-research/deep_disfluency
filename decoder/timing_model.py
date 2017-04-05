# To affect an HSMM, we want the probability of the state given the length of
# time to the last word
# This is only for non-0 transitions in the HMM


# STEP 1 GET ALL THE TIMINGS FOR EACH WORD
import numpy as np
import time
import sys
import theano
from collections import defaultdict
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import cPickle


from load import load
from load.load import load_data_from_array, load_tags
from load.load import load_increco_data_from_file, load_data_from_timings_file

full_label2idx = load_tags("../data/tag_representations/swbd1_trp_tags.csv")
full_idx2label = dict((v, k) for k, v in full_label2idx.iteritems())


def load_data(divisions, range_files, simple=False):
    # make the args into a dict
    s = dict()
    s['acoustic'] = True
    s['window'] = 2
    s["tags"] = '1_trp_simple'
    s['train_data'] = 'swbd1_train'
    s['use_saved_model'] = False
    print s
    if s['acoustic']:
        s['acoustic'] = 350  # dimension of acoustic features vector
    pre_load_training = True
    print "loading data and tag sets"
    _, _, _, train_dict = \
        load.switchboard_data(train_data=s['train_data'], tags=s['tags'])

    # get relevant dictionaries
    idx2label = dict((v, k) for k, v in train_dict['labels2idx'].iteritems())
    idx2word = dict((v, k) for k, v in train_dict['words2idx'].iteritems())
    if train_dict.get('pos2idx'):
        idx2pos = dict((v, k) for k, v in train_dict['pos2idx'].iteritems())

    if simple:
        overall_simple_trp_label2idx = {0: "<cc>",
                                        1: "<ct>",
                                        2: "<tc>",
                                        3: "<tt>"}
        simple_trp_label2idx = {0: "<c",
                                1: "<t"}
        simple_trp_label2idx2 = {0: "c>",
                                 1: "t>"}
    else:
        simple_trp_label2idx = train_dict['labels2idx']
    # we will shuffle the train dialogues, keep heldout and test non-scrambled
    data = defaultdict(list)
    # a map from the label index to every duration of that label
    timing_dict = defaultdict(list)
    timing_dict2 = defaultdict(list)
    for key, rangefile in zip(divisions, range_files):
        file_ranges = [line.strip("\n") for line in open(rangefile)]
        if s['use_saved_model']:
            if "train" in key:
                continue
            if "asr" in key:
                corpus = key[:key.find("_")]
                corpus = corpus[0].upper() + corpus[1:]
                increco_file = "../data/asr_results/SWDisf{}_\
                    pos_increco.text".format(corpus)
                data[key] = load_increco_data_from_file(
                    increco_file,
                    train_dict['words2idx'],
                    train_dict['pos2idx'])

            else:
                f_path = "../data/disfluency_detection/switchboard/swbd_{}_\
                    partial_timings_data.csv".format(key)
                data[key] = load_data_from_timings_file(
                    f_path, train_dict['words2idx'], train_dict['pos2idx'])
            continue
        for f in file_ranges:
            for part in ["A", "B"]:
                dialogue_speaker_data = "/media/dsg-labuser/NO_NAME/\
                    simple_rnn_disf_data/audio/" + f + part + ".npy"
                if key != "train" or pre_load_training:
                    print "loading", dialogue_speaker_data
                    dialogue_speaker_data = np.load(dialogue_speaker_data)
                    dialogue_speaker_data = load_data_from_array(
                        dialogue_speaker_data,
                        n_acoust=s['acoustic'],
                        cs=s['window'],
                        tags=s["tags"],
                        full_idx_to_label_dict=full_idx2label)
                    # print len(dialogue_speaker_data)
                    timings = dialogue_speaker_data[0]
                    # words = dialogue_speaker_data[2]
                    # pos = dialogue_speaker_data[3]
                    labels = dialogue_speaker_data[-1]
                    # print timings, words, labels
                    prev = 0
                    prev_t = 0
                    prev_prev_t = 0  # effecting a trigram of timings
                    if "train" not in key:
                        simple_trp_label2idx = overall_simple_trp_label2idx
                    for i in range(0, len(timings)):
                        t = timings[i]-prev
                        complex_tag = idx2label[labels[i]]
                        found = False
                        tag = labels[i]
                        if simple:
                            for k, v in simple_trp_label2idx.items():
                                if v in complex_tag:
                                    tag = k
                                    found = True
                                    break
                            if not found:
                                raw_input("warning")
                        if t < 0:
                            print "below zero"
                            t = np.average([x[0] for x in timing_dict[tag]])
                            timings[i-1] = timings[i] - t
                        timing_dict[tag].append((prev_prev_t, prev_t, t))
                        # now the second one
                        if "train" in key:
                            found = False
                            if simple:
                                for k, v in simple_trp_label2idx2.items():
                                    if v in complex_tag:
                                        tag = k
                                        found = True
                                        break
                                if not found:
                                    raw_input("warning")
                            timing_dict2[tag].append((prev_prev_t, prev_t, t))
                        prev = timings[i]
                        prev_prev_t = prev_t
                        prev_t = t
                data[key].append((f + part, dialogue_speaker_data))
    X = []
    y = []
    for i in sorted(timing_dict.keys()):
        print simple_trp_label2idx[i]
        print np.average([time[0] for time in timing_dict[i]]),
        print np.std([time[0] for time in timing_dict[i]])
        for tup in timing_dict[i]:
            X.append(list(tup))
            y.append(i)
    X = np.asarray(X)
    y = np.asarray(y)
    print X.shape
    print y.shape
    if len(timing_dict2.keys()) > 0:
        X2 = []
        y2 = []
        for i in sorted(timing_dict2.keys()):
            print simple_trp_label2idx2[i]
            print np.average([time[2] for time in timing_dict2[i]])
            print np.std([time[0] for time in timing_dict2[i]])
            for tup in timing_dict2[i]:
                X2.append(list(tup))
                y2.append(i)
        X2 = np.asarray(X2)
        y2 = np.asarray(y2)
        labels = [simple_trp_label2idx2[k] for k in sorted(
            simple_trp_label2idx2.keys())]
        print labels
    else:
        X2 = []
        y2 = []
    return X, y, X2, y2


def train(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = LogisticRegression(class_weight='auto')
    model.fit(X, y)

    print(model)
    return model, scaler


def test_simple(model, scaler, data, y):
    # make predictions
    X = scaler.transform(data)
    expected = y
    print expected
    predicted = model.predict(X)
    print metrics.classification_report(expected, predicted)
    print metrics.confusion_matrix(expected, predicted)


def test(model, scaler, X, y, model2, scaler2):
    # make predictions
    def convert_to_overall_dict(a, b):
        if a == 0:
            if b == 0:
                return 0
            elif b == 1:
                return 1
        elif a == 1:
            if b == 0:
                return 2
            elif b == 1:
                return 3
        print a, b, "wrong!"
        return None

    def convert_to_two_singles(a):
        if a == 0:
            return 0, 0
        if a == 1:
            return 0, 1
        if a == 2:
            return 1, 0
        if a == 3:
            return 1, 1
    test1 = []
    test2 = []
    for x in list(y):
        a, b = convert_to_two_singles(x)
        test1.append(a)
        test2.append(b)

    X1 = scaler.transform(X)
    X2 = scaler2.transform(X)
    expected = y

    predicted = model.predict(X1)
    print metrics.classification_report(np.asarray(test1), predicted)
    print metrics.confusion_matrix(np.asarray(test1), predicted)

    predicted2 = model2.predict(X2)
    print metrics.classification_report(np.asarray(test2), predicted2)
    print metrics.confusion_matrix(np.asarray(test2), predicted2)
    overallpredicted = []
    for p in range(0, len(list(predicted))):
        overallpredicted.append(convert_to_overall_dict(predicted[p],
                                                        predicted2[p]))

    predicted = np.asarray(overallpredicted)
    # summarize the fit of the model
    print metrics.classification_report(expected, predicted)
    print metrics.confusion_matrix(expected, predicted)

range_dir = "../data/disfluency_detection/swda_divisions_disfluency_detection/"
X, y, X2, y2 = load_data(divisions=["train"], range_files=[
                    range_dir + "SWDisfTrainWithAudio_ranges.text",
                    range_dir + "SWDisfHeldout_ranges.text",
                    range_dir + "SWDisfHeldout_ranges.text",
                    range_dir + "SWDisfTest_ranges.text",
                    range_dir + "SWDisfHeldoutASR_ranges.text",
                    range_dir + "SWDisfTestASR_ranges.text"
                    ], simple=True)
model, scaler = train(X, y)
model2, scaler2 = train(X2, y2)


X, y, X2, y2 = load_data(divisions=["heldout"], range_files=[
                   range_dir + "SWDisfHeldout_ranges.text",
                   range_dir + "SWDisfTest_ranges.text",
                   range_dir + "SWDisfHeldoutASR_ranges.text",
                   range_dir + "SWDisfTestASR_ranges.text"
                   ], simple=True)
test(model, scaler, X, y, model2, scaler2)
# save the classifier
with open('LogReg_balanced_timing_classifier.pkl', 'wb') as fid:
    cPickle.dump(model, fid)
with open('LogReg_balanced_timing_scaler.pkl', 'wb') as fid:
    cPickle.dump(scaler, fid)
