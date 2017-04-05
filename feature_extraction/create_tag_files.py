"""Methods for: 

--creation of vocabulary files from training corpora with index for each word
--creation of files with the disfluency tags and their index.
"""
import argparse
import re
from collections import defaultdict

from feature_utils import load_data_from_corpus_file
from feature_utils import load_data_from_disfluency_corpus_file
from feature_utils import get_tags
from copy import deepcopy

def concat_all_data_all_speakers(dialogues):
    frames = []
    words = []
    pos = []
    indices = []
    labels = []
    for d in dialogues:
        _, data = d
        frames_data, lex_data, pos_data, indices_data,labels_data = data
        frames.append(deepcopy(frames_data))
        words.append(deepcopy([x[0][1] for x in lex_data]))
        pos.append(deepcopy([x[1] for x in pos_data]))
        indices.append(deepcopy(indices_data))
        labels.append(deepcopy(labels_data))
    print "concatenated all data"
    #print [len(x) for x in [frames, words, pos, indices, labels]]
    return frames, words, pos, indices, labels

def create_word_or_pos_representation(wordrepfilepath,
                                      words_or_pos,
                                       min_occurrences=1, 
                                       word_rep_type="one-hot"):
    """Creates a word rep file from a training data file supplied or a 
    plain text file. Has to occur at least min_occurrences times 
    else set to the unknown token automatically externally.
    """
    tag_dict = defaultdict(int) #tag and the number of times it occur
    
    print "creating word or pos tag file..."
    i = 0
    for tag_sequence in words_or_pos:
        i+=1
        #print i
        #print len(tag_sequence)
        for a_tag in tag_sequence:
            tag_dict[a_tag]+=1
    vocab_allowed =  [key for key in tag_dict.keys() if 
                      tag_dict[key]>=min_occurrences]
    vocab = [str(i)+","+str(vocab_allowed[i]) 
             for i in range(0,len(vocab_allowed))]
    tagstring = "\n".join(vocab)
    
    print "vocab_size" , len(vocab)
    myfile = open(wordrepfilepath,"w")
    myfile.write(tagstring)
    myfile.close()
    print "word or pos tag file complete."


def create_tag_representations(tag_rep_filepath, tags, 
                                        representation="disf1", limit=8):
    """Create the tag files for a given corpus f with a given 
    representation type.
    
    Keyword arguments:
    tag_rep_file --  file to write the tag rep too.
    Note this must be a training file.
    tags -- list of lists of tags (training)
    representation -- string showing the type of tagging system, 
    1=standard, 2=rm-N values where N does not count intervening edit terms
    3=same as 2 but with a 'c' tag after edit terms have ended.
    """
    tag_dict = defaultdict(int) #tag and the number of times it occurs
    print "creating tag file:", representation, "..."
    #print len(tags)
    i = 0
    for tag_sequence in tags:
        i+=1
        #print i
        #print len(tag_sequence)
        for a_tag in tag_sequence:
            tag = ""
            subtags = get_tags(a_tag)
            if "disf" in representation:
                for t in subtags:
                    if not re.search(r'<[ct]*/>',t)\
                    and not re.search(r'<diact type=".*"/>',t)\
                    and not re.search(r'<speechLaugh/>|<laughter/>',t):
                        tag+=t
            if "uttseg" in representation:
                m = re.search(r'<[ct]*/>',a_tag)
                if m: tag+=m.group(0)
            if "dact" in representation:
                m = re.search(r'<diact type=".*"/>',a_tag)
                if m: 
                    tag+=m.group(0)
                #elif not "<laughter" in a_tag:
                #    print "warning no diact!",a_tag
                    #TODO laughter as a diact or not?
            if "laugh" in representation:
                m = re.search(r'<speechLaugh/>|<laughter/>',a_tag)
                if m: 
                    tag+=m.group(0)
                else:
                    tag+="<nolaughter/>"
            if tag == "":
                if not "<laugh" in a_tag:
                    print "warning no tag", a_tag
                continue
            tag_dict[tag]+=1    
        
    print tag_dict
    tagstring = "\n".join([str(i)+","+str(sorted(tag_dict.keys())[i]) 
                           for i in range(0,len(tag_dict.keys()))])
    tag_rep_file = open(tag_rep_filepath,"w")
    tag_rep_file.write(tagstring)
    tag_rep_file.close()
    print "tag file complete."
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates tag sets for \
                                     labels.')
    parser.add_argument('-i', action='store', dest='corpusFile', 
                        default='../data/disfluency_detection/\
                        switchboard/swbd_disf_train_1_partial_data.csv', 
                        help='location of the corpus from which to\
                        generate the tag sets')
    parser.add_argument('-tag', action='store', dest='tagFolder',
                        default=None,
                        help='location to store the tag files to\
                        tag index mapping')
    parser.add_argument('-u', action='store_true', dest='uttSeg',
                        default=False,
                        help='Whether to annotate with utterance segmentation\
                        tags.')
    parser.add_argument('-d', action='store_true', dest='dialogueActs',
                        default=False,
                        help='Whether to annotate with dialogue acts.')
    parser.add_argument('-l', action='store_true', dest='laughter',
                        default=False,
                        help='Whether to annotate with laughter.')
    parser.add_argument('-joint', action='store_true', dest='joint',
                        default=False,
                        help='Whether to create a joint tag set with the \
                        cross product of the tags (which appear in the data.')
    args = parser.parse_args()
    
    corpus_file = args.corpusFile
    print "creating tag files. Using file:", corpus_file
    word_rep_file = args.tagFolder + "/swbd_word_rep.csv"
    pos_rep_file =  args.tagFolder + "/swbd_pos_rep.csv"
    disf_tag_rep_file = args.tagFolder + "/swbd_disf1_tags.csv"
    joint_tag_rep_file = disf_tag_rep_file
    
    #input_file_type -- timings|disf whether a timings file or plain
    input_file_type = "timings" 
    if input_file_type == "timings":
        dialogues = load_data_from_corpus_file(corpus_file)
        _, words, pos, _, labels = concat_all_data_all_speakers(dialogues)
    else:
        _, _, words, pos, labels = load_data_from_disfluency_corpus_file(
                                                                corpus_file)
    create_tag_representations(disf_tag_rep_file,labels, 
                               representation="disf1")
     
    if args.uttSeg:
        joint_tag_rep_file = joint_tag_rep_file.replace("_tags",
                                                  "_uttseg_tags")
        #doing joint one here
        uttseg_tag_rep_file = disf_tag_rep_file.replace("disf1_tags",
                                                  "uttseg_tags")
        create_tag_representations(uttseg_tag_rep_file, labels, 
                               representation="uttseg")
        uttseg_disf_tag_rep_file = disf_tag_rep_file.replace("_tags",
                                                  "_uttseg_tags")
        create_tag_representations(uttseg_disf_tag_rep_file, labels, 
                               representation="disf1_uttseg")
     
    if args.dialogueActs:
        joint_tag_rep_file = joint_tag_rep_file.replace("_tags",
                                                  "_dacts_tags")
        dact_rep_file = uttseg_tag_rep_file.replace("_tags",
                                                  "_dacts_tags")
        create_tag_representations(dact_rep_file, labels,
                               representation="uttseg_dact")
    if args.laughter:
        joint_tag_rep_file = joint_tag_rep_file.replace("_tags",
                                                  "_laughter_tags")
        laughter_rep_file = disf_tag_rep_file.replace("_disf1_tags",
                                                  "_laughter_tags")
        create_tag_representations(laughter_rep_file, labels, 
                                   representation="laughter")
    if args.joint:
        create_tag_representations(joint_tag_rep_file, labels,
                        representation="disf1_uttseg_dacts_laughter")
    
    create_word_or_pos_representation(word_rep_file, words,
                                      min_occurrences=2)
    create_word_or_pos_representation(pos_rep_file, pos,
                                      min_occurrences=1)