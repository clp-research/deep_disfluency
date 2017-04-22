
# coding: utf-8

# In[1]:

#Extract features from the audio feature files and save them as numpy arrays

#TODO
#speed tests for loading in the audio data and getting the quickest way of loading in the data each epoch
#though in theory this should be possible on the GPU as a shared variable. Options:
#1. Load each dialogue as a pickled numpy object 'ready to go' as a big old matrix 
#the words will only appear when their end time is reached.
#in the event of 0-length words (hopefully there won't be any- check online), approximate length in terms of X 10ms frames/letter
#find X from corpus
#2. Load each dialogue as an individual CSV file with all the features in columns
#will be identical to the pickled numpy object and just requires reading in- this may be slower. Can use this for 
#sanity check and plotting later on during experiments.


# In[2]:

from copy import deepcopy
import os
import math
from feature_utils import load_data_from_disfluency_corpus_file, sort_into_dialogue_speakers
import sys
sys.path.append("../")
sys.path.append("../../")
from deep_disfluency.load.load import load_word_rep, load_tags
import pandas
import numpy as np


# In[3]:


# In[4]:

def open_with_pandas_read_csv(filename,header=None,delim="\t"):
    df = pandas.read_csv(filename, sep=delim,header=header,comment='f') #specific to these files
    data = df.values
    return data  


# In[5]:

def myround(x, base=.01,prec=2):
    return round(base * round(float(x)/base),prec)


# In[6]:

def get_interval_indices(word_file, ms, context_frames):
    """For a given list of start and end times for intervals of interest (which will be words),
    return a matrix of start_time/stop times rounded up to the nearest window_size ms.
    """
    data = open_with_pandas_read_csv(word_file)[:,1]
    #print data
    #print data.shape
    #first, round up the time to the nearest int in terms of frames
    final_data = []
    for x in data:
        stop = int(100.0 * x)
        assert stop - (stop - context_frames) == 50, stop - (stop - context_frames)
        final_data.append((stop - context_frames, stop))
    return final_data


# In[7]:

def get_audio_features(filename,features=[0,2,3,4,5,6,7,8],interval_indices=None):
    """For a given file, extract the features at the indices specified for given
    the given intervals- return a list of tuples (end_time, array_of_features_for_that_interval) 
    the list will be the same length as the number of intervals """
    "frameIndex; frameTime; pcm_RMSenergy_sma; pcm_LOGenergy_sma; F0final_sma;    voicingFinalUnclipped_sma; F0raw_sma; pcm_intensity_sma; pcm_loudness_sma"
    #print intervals
    print filename
    data = None

    data = open_with_pandas_read_csv(filename,header=True,delim=";")
    final_data = []
    
    data = data[:,features] #just get the features of interest
    #print data[0]
    
    for interval in interval_indices:
        start, stop = interval
        #print start, stop
        #print data[start-1:stop-1]
        #raw_input()
        my_data = data[start:stop] #to account for the header
        #print my_data
        #print my_data.shape
        if start < 0 and my_data.shape[0] < 50:
            print "beneath 0 starting context, add padding"
            padding = np.zeros((0-start,data.shape[1]))
            my_data = np.concatenate([padding,my_data])
            #raw_input()
        if my_data.shape[0] < 50:
            print "adding end padding"
            padding = np.zeros((50-my_data.shape[0],data.shape[1]))
            my_data = np.concatenate([my_data,padding])
        assert my_data.shape[0] == 50, my_data.shape[0]
        final_data.append(my_data)
    return final_data


# In[8]:

def main(wordtiming_dir, audiofeatures_dir, target_dir):

    if not os.path.exists(target_dir): 
        os.mkdir(target_dir)
    
    
    range_dir = "../data/disfluency_detection/swda_divisions_disfluency_detection/"
    
    range_files = [range_dir + "SWDisfTrainWithAudio_ranges.text",
                   range_dir + "SWDisfHeldout_ranges.text",
                   range_dir + "SWDisfTest_ranges.text"
                   ]
    ranges = {}
    for key, rangefile in zip(["train", "heldout", "test"], range_files):
        file_ranges = [line.strip("\n") for line in open(rangefile)]
        ranges[key] = sorted(deepcopy(file_ranges))
    for key, value in ranges.items():
        print key, len(value)
    
    
    
    #split the big disfluency marked -up files into individual file tuples
    #it is possible to do the matching on the utterance level as they should have consistent mark-up between the two
    disf_dir = "../data/disfluency_detection/switchboard"
    disfluency_files = [disf_dir+"/swbd1_train_partial_data.csv",
                        disf_dir+"/swbd_heldout_partial_data.csv",
                        disf_dir+"/swbd_test_partial_data.csv"]
    dialogue_speakers = []
    for key, disf_file in zip(["train", "heldout", "test"],disfluency_files):
        IDs, mappings, utts, pos_tags, labels = load_data_from_disfluency_corpus_file(disf_file)
        dialogue_speakers.extend(sort_into_dialogue_speakers(IDs,mappings,utts, pos_tags, labels))
    word_pos_data = {} #map from the file name to the data
    for data in dialogue_speakers:
        dialogue,a,b,c,d = data
        word_pos_data[dialogue] = (a,b,c,d)
    
    
    # In[9]:
    
    DATAPREFIX = "../data"
    tags = "1_trp"
    
    test_dict = {}
    test_dict['words2idx'] = load_word_rep("/".join([DATAPREFIX,'tag_representations','swbd_word_rep.csv']))
    test_dict['pos2idx'] = load_word_rep("/".join([DATAPREFIX,'tag_representations', 'swbd_pos_rep.csv']))
    test_dict['labels2idx'] = load_tags("/".join([DATAPREFIX,'tag_representations','swbd'+tags+'_tags.csv']))
    
    word_dict = test_dict['words2idx']
    pos_dict = test_dict['pos2idx']
    label_dict = test_dict['labels2idx']
    
    
    # In[10]:
    
    #We're going to make the data bundle for the experiments here
    #we will make all the entries for each dialogue. These can get shuffled at test time, but the important thing is
    #that they're fast to load dynamically one-by-one (there will be 950 trainig/ 104 ho, 102 test roughly)
    #if np is very fast, this is fine, if it isn't we might need to load in folds of say 100 each time,
    #let's test it
    #the line of every file/numpy pickle is
    #end_of_word_time, f1window(500ms), f2window.... fnwindow(500ms), wordindex, posindex, labelindex 
    missed = []
    for d in ranges['train'] + ranges['heldout'] + ranges['test']:
        for part in ["A",'B']:
            all_features = []
            print d,part
            #if int(d) < 4005:
            #    continue
            test = wordtiming_dir + "/{}{}.csv".format(d,part)
            ind = get_interval_indices(test, 0.01, 50)
    
            test_features = audiofeatures_dir + "/sw0{}{}.csv".format(d,part)
            audio = None
            try:
                audio = get_audio_features(test_features,interval_indices=ind)
            except IOError:
                missed.append(test_features)
                continue
            _,words,pos,labels = word_pos_data[d+part]
            #print len(audio), len(words)
            #break
            for i in range(0,len(audio)):
                word_ix = word_dict.get(words[i])
                if word_ix == None:
                    word_ix = word_dict.get("<unk>")
                #print word_ix
                pos_ix = pos_dict.get(pos[i])
                if pos_ix == None:
                    pos_ix = pos_dict.get("<unk>")
                #print pos_ix
                #print labels[i]
                label_ix = label_dict.get(labels[i])
                if label_ix == None:
                    print "no label for", labels[i]
                    raise Exception
                #print label_ix
                #print audio[i].shape
                #print audio[i][:,1:].shape
                audio_d1 = audio[i][:,1:].shape[0]
                audio_d2 = audio[i][:,1:].shape[1] #don't want the first timeindex column?
                final_audio = audio[i][:,1:].reshape((audio_d1 * audio_d2, 1))   #flattened now
                #print final_audio.shape
                final_lexical =  np.asarray([[ word_ix, pos_ix, label_ix ]]).reshape((3,1))
                #print final_lexical.shape
                final_end_word_time =  np.asarray([[audio[i][:,0][-1]]])
                #print final_end_word_time.shape
                final_vector = np.concatenate([final_end_word_time, final_audio, final_lexical])
                #print final_vector.shape
                #print final_vector
                #print final_vector[-3:]
                #raw_input()
                #if y == "q": break
                all_features.append(final_vector)
            
            allnumpy = np.concatenate(all_features,axis=1)
            #file = open(data_dir+d+part+".csv","w")
            #for i in all_features:
            #    file.write("/t".join([str(x) for x in i])+"\n")
            #file.close()
            #print allnumpy.shape
            np.save(target_dir+"/"+d+part+".npy",allnumpy)
            #break #TODO remove in real
        #break #TODO remove in real
    print missed

# In[ ]:
if __name__ == '__main__':
    #NOTE you need to give the locations of the data relative to this file
    rootdir = "../../../swbd"
    wordtiming_dir = rootdir + "/mapping_MS2SWDA"
    audiofeatures_dir = rootdir + "/audio_features"
    target_dir = "../data/audio"
    main(wordtiming_dir, audiofeatures_dir, target_dir)