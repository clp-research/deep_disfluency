from __future__ import division
#simple script to take the .npy files with the acoustic data and convert them into seconds
#and map them to the disfluency detection files and create a new file with the Timings
#TODO we're going to scrap this and make it possible to do it as
#a method from feature_utils.py
#

#First load the data
import theano
import numpy as np
from collections import defaultdict
import sys
from copy import deepcopy
from feature_utils import get_tag_data_from_corpus_file, sort_into_dialogue_speakers, wer

sys.path.append('../../') #path to the src files

from rnn_disf_detection.load import load
from rnn_disf_detection.rnn.elman import Elman
from rnn_disf_detection.rnn.lstm import Lstm
from rnn_disf_detection.utils.tools import shuffle, minibatch, contextwin, contextwinbackwards, corpus_to_indexed_matrix, convertFromIncDisfluencyTagsToEvalTags
from rnn_disf_detection.utils.accuracy import save_to_disfeval_file
#from rnn_disf_detection.evaluation.disf_evaluation import vec_tags_from_text_file, disfluency_accuracy
from rnn_disf_detection.embeddings.load_embeddings import populate_embeddings
#from rnn_disf_detection.disf_decoder.hmm import rnn_hmm
from rnn_disf_detection.load.load import load_data_from_array, load_tags
from feature_utils import process_arguments, fill_in_time_approximations

theano.config.optimizer='None' #speeds things up marginally

full_label2idx = load_tags("../data/tag_representations/swbd1_trp_tags.csv") #NB mainly for experiments
full_idx2label = dict((k,v) for v,k in full_label2idx.iteritems()) # fir

def run_experiment(args):
    #make the args into a dict
    s = dict()
    for feat,val in args._get_kwargs():
        s[feat] = val
        #print feat,val
    print s
    #raw_input()
    
    if s['acoustic']:
        s['acoustic']  = 350 #dimension of acoustic features vector (essentially 7 * 50)
        
    pre_load_training = True # TODO if not on Bender this is quite big so can't load training- switch to true on Bender
    
    print "loading data and tag sets" #NB Don't get the tag sets here directly anymore
    _, _, _, train_dict = load.switchboard_data(train_data=s['train_data'],tags=s['tags'])
    
    #get relevant dictionaries
    idx2label = dict((k,v) for v,k in train_dict['labels2idx'].iteritems()) # first half (28) the same as the test
    idx2word  = dict((k,v) for v,k in train_dict['words2idx'].iteritems())
    if not train_dict.get('pos2idx') == None:
        idx2pos = dict((k,v) for v,k in train_dict['pos2idx'].iteritems())

    range_dir = "../data/disfluency_detection/swda_divisions_disfluency_detection/"
    range_files = [
                   #range_dir + "SWDisfTrainWithAudio_ranges.text",
                   range_dir + "SWDisfHeldout_ranges.text",
                   range_dir + "SWDisfTest_ranges.text"
                   ]
    
    #get the filenames for the train/heldout/test division
    data = defaultdict(list) #we will shuffle the train dialogues, keep heldout and test non-scrambled:
    divisions =  ["heldout", "test"]
    if not s['use_saved_model']:
        divisions =  ["train", "heldout", "test"]
        
    for key, rangefile in zip(divisions, range_files):
        file_ranges = [line.strip("\n") for line in open(rangefile)]
        for f in file_ranges:
            for part in ["A","B"]:
                dialogue_speaker_data = "../data/audio/" + f + part + ".npy"
                #if we're pre-loading up data, which we do anyway for the test/heldout
                #but also for training when using a big machine
                if key != "train" or pre_load_training:
                    print "loading",dialogue_speaker_data
                    dialogue_speaker_data = np.load(dialogue_speaker_data)
                    dialogue_speaker_data = load_data_from_array(dialogue_speaker_data,
                                                                 n_acoust=s['acoustic'],
                                                                 cs=s['window'],
                                                                 tags=s["tags"],
                                                                 full_idx_to_label_dict=full_idx2label)
                data[key].append((f + part, dialogue_speaker_data)) #tuple of name + data/file location
    
    #Now load the normal disfluency files with lexical + pos data
    disf_dir = "../data/disfluency_detection/switchboard"
    disfluency_files = [
                        disf_dir+"/swbd_heldout_partial_data.csv",
                        disf_dir+"/swbd_test_partial_data.csv"
                        ]
    dialogue_speakers = []
    for key, disf_file in zip(["heldout", "test"],disfluency_files):
        IDs, mappings, utts, pos_tags, labels = get_tag_data_from_corpus_file(disf_file)
        dialogue_speakers.extend(sort_into_dialogue_speakers(IDs,mappings,utts, pos_tags, labels))
    word_pos_data = {} #map from the file name to the data
    for new_data in dialogue_speakers:
        dialogue,a,b,c,d = new_data
        word_pos_data[dialogue] = (a,b,c,d)
    
    #Now combine the two into super file of the following format
    example = """"
    File: KB3_1
    KB3_1:1    0.00    1.12    $unc$yes    NNP    <rms id="1"/><tc/>
    KB3_1:2    1.12    2.00     $because    IN    <rps id="1"/><cc/>
    KB3_1:3    2.00    3.00    because    IN    <f/><cc/>
    KB3_1:4    3.00    4.00    theres    EXVBZ    <f/><cc/>
    KB3_1:6    4.00    5.00    a    DT    <f/><cc/>
    KB3_1:7    6.00    7.10    pause    NN    <f/><cc/>"""
    
    unfound = []
    found = []
    for div, disf_file in zip(divisions, disfluency_files):
        print div
        newfile = open(disf_file.replace("_data","_timings_data"),"w")
        for speaker in word_pos_data.keys():
            
            #print "speaker", speaker
            frames_data = None
            foundbool = False
            for convo in data[div]:
                #print convo[0]
                if convo[0] == speaker:
                    frames_data = convo[1][0]
                    foundbool = True
                    break
            if foundbool == False:
                #raw_input()
                if not speaker in found:
                    unfound.append(speaker)
                continue
            newfile.write("File: " + speaker+"\n") #just for the main divisions train/heldout/test
            found.append(speaker)
            if speaker in unfound:
                unfound.remove(speaker)
            timing_data = [(0,frames_data[0])]
            for x in range(1,len(frames_data)):
                timing_data.append((frames_data[x-1],frames_data[x]))
            final_tags = convertFromIncDisfluencyTagsToEvalTags(deepcopy(word_pos_data[speaker][3]),deepcopy(word_pos_data[speaker][1]),start=0,representation="1_trp")
            lengths = [len(word_pos_data[speaker][0]), len(word_pos_data[speaker][1]), len(word_pos_data[speaker][2]),len(timing_data), len(final_tags)]
            #print lengths
            assert(all(x == lengths[0] for x in lengths)),lengths
            #correct the timings for missing values
            timing_data = [(a,b) for a,b,c in fill_in_time_approximations([(x, y[0],y[1]) for x, y in zip(word_pos_data[speaker][1],timing_data)])]
            for ref,timing,word,pos,tag in zip(word_pos_data[speaker][0],timing_data,word_pos_data[speaker][1],word_pos_data[speaker][2],final_tags):
                start = 0.0
                if timing[0]>0:
                    start = float(timing[0]/100) #convert to seconds
                end = 0.0
                if timing[1]>0:
                    end = float(timing[1]/100)
                assert end>=start,'End before start {} {} in file {}'.format([start,end,speaker])
                newfile.write("\t".join([ref,str(start),str(end),word,pos,tag])+"\n")
            newfile.write("\n")
        newfile.close()
    print len(unfound), "unfound", unfound
        
if __name__ == '__main__':
    #args = process_arguments(config='experiment_configs.csv',exp_id='034',use_saved=37,hmm=False)
    #run_experiment(args)
    args = process_arguments(config='../experiments/experiment_configs.csv', exp_id='034', use_saved=37, hmm=True)
    run_experiment(args)
    #for exp in range(31,40):
    #    args.exp = exp
    #    run_experiment(args) #could in theory try multiple instances for multiple experiments