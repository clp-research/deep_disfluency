
# coding: utf-8

# In[1]:

from __future__ import division
#from nltk.tag.perceptron import PerceptronTagger
from copy import deepcopy
from nltk.tag import CRFTagger


# In[2]:

from collections import Counter


# In[3]:

from feature_utils import load_data_from_disfluency_corpus_file, sort_into_dialogue_speakers, wer


# In[4]:

#tagger = PerceptronTagger(load="swbd_postagger")
tagger = ct = CRFTagger()
ct.set_model_file("crfpostagger")


# In[5]:

disf_dir = "../data/disfluency_detection/switchboard"
disfluency_files = [
                    disf_dir+"/swbd_heldout_partial_data.csv",
                    ]
dialogue_speakers = []
for key, disf_file in zip(["heldout", "test"],disfluency_files):
    IDs, mappings, utts, pos_tags, labels = load_data_from_disfluency_corpus_file(disf_file)
    dialogue_speakers.extend(sort_into_dialogue_speakers(IDs,mappings,utts, pos_tags, labels))
word_pos_data = {} #map from the file name to the data
for data in dialogue_speakers:
    dialogue,a,b,c,d = data
    word_pos_data[dialogue] = (a,b,c,d)


# In[6]:

ct.tag([unicode(w) for w in "uh my name is john".split()])


# In[ ]:

#either gather training data or test data
training_data = []
for speaker in word_pos_data.keys():
    #print speaker
    sp_data = []
    prefix = []
    predictions = []
    for word,pos in zip(word_pos_data[speaker][1],word_pos_data[speaker][2]):
        prefix.append(unicode(word.replace("$unc$","").encode("utf8")))
        prediction = ct.tag(prefix[-5:])[-1][1]
        sp_data.append((unicode(word.replace("$unc$","").encode("utf8")),unicode(pos.encode("utf8"))))
        predictions.append(prediction)
    #predictions = tagger.tag(prefix)
    training_data.append(deepcopy([(r,h) for r,h in zip(predictions,sp_data)]))
    #training_data.append(deepcopy(sp_data))


# In[ ]:

#if training a new crf tagger, uncomment
#tagger.train(training_data,"crfpostagger")
#x = "h"
#if x == "h" : quit()


# In[ ]:

tp = 0
fn = 0
fp = 0
overall_tp = 0
overall_count = 0
c = Counter()
for t in training_data:
    for h,r in t:
        #print h,r
        overall_count+=1
        hyp = h
        if hyp == "UH":
            if not r[1] == "UH":
                fp+=1
            else:
                #print h,r
                tp+=1
        elif r[1] == "UH":
            
            fn+=1
        if hyp == r[1]: 
            overall_tp+=1
        else:
            c[hyp + "-" + r[1]]+=1
            
   # raw_input()
print tp, fn, tp
p = (tp/(tp+fp))
r = (tp/(tp+fn))
print "UH p, r, f=", p, r, (2 * p * r)/(p+r)
print "overall accuracy", overall_tp/overall_count


# In[ ]:

print "most common errors hyp-ref", c.most_common()[:20]


# In[7]:

#now tag the incremental ASR result files of interest
asr_dir = "../data/asr_results/"
inc_asr_files = [asr_dir + "SWDisfTest_increco.text", asr_dir + "SWDisfHeldout_increco.text"]


# In[18]:

def get_diff_and_new_prefix(current,newprefix,verbose=False):
    """Only get the different right frontier according to the timings
    and change the current hypotheses"""
    if verbose: 
        print "current", current
        print "newprefix", newprefix
    rollback = 0
    original_length = len(current)
    for i in range(len(current)-1,-2,-1):
        if verbose: print "oooo", current[i], newprefix[0]
        if i==-1 or (float(newprefix[0][1]) >= float(current[i][2])):
            if i==len(current)-1:
                current = current + newprefix
                break
            k = 0
            marker = i+1
            for j in range(i+1,len(current)):
                if k == len(newprefix):
                    break
                if verbose: print "...", j, k, current[j], newprefix[k], len(newprefix)
                if not current[j]==newprefix[k]:
                    break
                else:
                    if verbose: print "repeat"
                    k+=1
                    marker = j+1
            rollback = original_length - marker   
            current = current[:marker] + newprefix[k:]
            newprefix = newprefix[k:]
            break
    return (current, newprefix, rollback)


# In[19]:

for filename in inc_asr_files:
    #always tag the right frontier
    current = []
    right_frontier = 0
    rollback = 0
    newfile = open(filename.replace("increco.","pos_increco."),"w")
    file = open(filename)
    dialogue = 0
    for line in file:
        if "File:" in line:
            dialogue = line
            newfile.write(line)
            current = []
            right_frontier = 0
            rollback = 0
            continue
        if "Time:" in line:
            increment = []
            newfile.write(line)
            continue
        
        #print "inc", increment
        if line.strip("\n") == "":
            if current == []:
                current = deepcopy(increment)
                #print "c", current
                #raw_input()
            else:
                verb = False
                #if "4074A" in dialogue:
                #    verb = True
                current, _, rollback = get_diff_and_new_prefix(deepcopy(current),deepcopy(increment),verb)
                #if "4074A" in dialogue:
                #    print "c", "r", "frontier", current, rollback, right_frontier
                #raw_input()
            for i in range(right_frontier - rollback,len(current)):
                #print i-4,i+1
                test = [unicode(x[0].lower().replace("'","")) for x in current[max([i-4,0]):i+1]]
                if "4074A" in dialogue:
                    print "test", test
                prediction = ct.tag(test)[-1][1]
                word = current[i][0].lower().replace("'","")
                if prediction in ["NNP","NNPS","CD","LS","SYM","FW"]:
                    word = "$unc$" + word
                start = current[i][1]
                end = current[i][2]
                newfile.write("\t".join([str(start),str(end),word] + [prediction]) + "\n")
            right_frontier = len(current)
            newfile.write(line)
        else:
            spl = line.strip("\n").split("\t")
            increment.append((spl[0],float(spl[1]),float(spl[2])))
    file.close()
    newfile.close()


# In[ ]:



