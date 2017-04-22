# To extract features from the raw data you will have to run the below in order.
# Not all of these may be required (e.g. not the Switchboard specific ones) and can be commented out.

# switchboard specific pre-processing. Make sure the swbd/ folder is a sister folder to this repo.
# In the swbd/ folder there needs to be 
#   - a /switchboard_audio/ subfolder as released by the LDC with the .sph files
#   - a /MSaligned/ subfolder which are the Mississippi time corrected transcripts
import argparse
import sys
import subprocess
import os

def extract_features(args):
        
    corpusName = args.divisionFile[args.divisionFile.rfind("/") + 1:].\
        replace("_ranges.text", "")
    if args.wordAlignmentFolder and 1 ==2:
        #link the word alignments to the disfluency detection corpora
        #also adds laughter
        command = [
            sys.executable,
            os.path.dirname(os.path.realpath(__file__)) + \
            '/swbd_map_word_alignments_to_SWDA_words.py',
            '-i', args.corpusLocation + "/" + corpusName + "_partial_data.csv",
            '-t', args.vectorFolder,
            '-f', args.divisionFile,
            '-a', args.wordAlignmentFolder
            ]
        if args.laughter:
            command.append('-l')
        subprocess.call(command)

    if args.newTags and 1 ==2:
        #create the tag representations (normally from the training data
        #not allowed to look into unseen tags in the test/dev set
        command = [
            sys.executable,
            os.path.dirname(os.path.realpath(__file__)) + \
                    '/create_tag_files.py',
            '-i', args.corpusLocation + "/" + corpusName + \
                    "_partial_data_timings.csv",
            '-tag', args.tagFolder,
            ]
        if args.laughter:
            command.append('-l')
        if args.uttSeg:
            command.append('-u')
        if args.dialogueActs:
            command.append('-d')
        if args.joint:
            command.append("-joint")
            
        subprocess.call(command)
    
    
    if args.ASR:
        pass
        #need to get ASR results for the corpus from the audio files
    
    if args.posTagger:
        if args.train_pos:
            pass
        pass
    
    if args.languageModelFolder:
        command = [
            sys.executable,
            os.path.dirname(os.path.realpath(__file__)) + \
                    '/add_language_model_features.py',
            '-i', args.corpusLocation + "/" + corpusName + \
            "_partial_data_timings.csv",
            '-lm', args.languageModelFolder,
            '-v', args.vectorFolder,
            '-order',str(3),
            '-xlm',
            #'-tag', args.tagFolder,
            ]
        if args.partial:
            command.append("-p")
        command.append('-e')
        if args.uttSeg:
            command.append('-u')
        print command
        subprocess.call(command)

    if args.audioFolder:
        pass

    extraction_results = {"POS_accuracy": None, "asr_WER": None}
    return extraction_results


if __name__ == '__main__':
    # NB to get the audio feature extraction to work you need to have
    # downloaded OpenSmile- see the README
    parser = argparse.ArgumentParser(description='Feature extraction for\
    disfluency and other tagging tasks from disfluency detection corpora and\
    raw data.')
    parser.add_argument('-i', action='store', dest='corpusLocation',
                        default='../data/disfluency_detection',
                        help='location of the disfluency\
                        detection corpus folder')
    parser.add_argument('-t', action='store', dest='vectorFolder',
                        default='../data/disfluency_detection/vectors',
                        help='location of the disfluency annotation csv files')
    parser.add_argument('-f', action='store', dest='divisionFile',
                        default='../data/disfluency_detection/\
                        swda_divisions_disfluency_detection/\
                        SWDisfTrain_ranges.text',
                        help='location of the file listing the \
                        files used in the corpus')
    parser.add_argument('-p', action='store_true', dest='partial',
                        default=False,
                        help='Whether to use partial words or not.')
    parser.add_argument('-a', action='store', dest='wordAlignmentFolder',
                        default=None,
                        help='location of the word alignment files')
    parser.add_argument('-tag', action='store', dest='tagFolder',
                        default=None,
                        help='location of the folder with the tag to\
                        tag index mapping')
    parser.add_argument('-new_tag', action='store_true', dest='newTags',
                        default=False,
                        help='Whether to save a new tag set generated from\
                        the data set to the tag folder.')
    parser.add_argument('-pos', action='store', dest='posTagger',
                        default=None, help='A POSTagger to tag the data.\
                        If None, Gold POS tags assumed.')
    parser.add_argument('-train_pos', action='store_true', dest='trainPOS',
                        default=False,
                        help='Whether to train POS a POS tagger on the data\
                        and save it at pos.')
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
    parser.add_argument('-lm', action='store', dest='languageModelFolder',
                        default=None,
                        help='Location of where to write a clean language\
                        model files out of this corpus.')
    parser.add_argument('-xlm', action='store_true',
                        dest='crossValLanguageModelTraining',
                        default=False,
                        help='Whether to use a cross language model\
                        training to be used for getting lm features on\
                        the same data.')
    parser.add_argument('-asr', action='store_true', dest='ASR',
                        default=False,
                        help='Whether to use IBM ASR to create ASR results.')
    parser.add_argument('-credentials', action='store', dest='credentials',
                        default="1841487c-30f4-4450-90bd-38d1271df295:\
                        EcqA8yIP7HBZ",
                        help="IBM Watson credentials of format username:pword")
    parser.add_argument('-audio', action='store', dest='audioFolder',
                        default=None,
                        help='location of the audio data \
                         files with .sph or wav files.')
    parser.add_argument('-opensmile', action='store', dest='openSmileConfig',
                        default=None,
                        help='location of the OpenSmile config file.')
    args = parser.parse_args()
    r = extract_features(args)
    print r