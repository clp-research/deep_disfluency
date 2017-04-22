"""
Script to run the experiments described in:
Julian Hough and David Schlangen.
Joint, Incremental Disfluency Detection and
Utterance Segmentation from Speech.
EACL 2017.

"""
import sys
import subprocess

# The data must been downloaded
# and put in place according to the top-level README
# each of the parts of the below can be turned off
# though they must be run in order so the latter stages work
create_disf_corpus = False
extract_features = False
train_models = False
test_models = True

asr = False  # extract and test on ASR results too


range_dir = "data/disfluency_detection/swda_divisions_disfluency_detection"
file_divisions_transcripts = [
	range_dir + "/swbd_disf_train_1_2_ranges.text",
	#range_dir + "/swbd_disf_train_audio_ranges.text", #audio features
	range_dir + "/swbd_disf_heldout_ranges.text",
	range_dir + "/swbd_disf_test_ranges.text",
							]
#the experiments in the EACL paper
#can just run the 'simple' tag sets
experiments = [
				33,
		#		34,
				35,
		#		36,
				37,
				38,
		#		39
			]

#1. Create the base disfluency tagged corpora in a standard format
if create_disf_corpus:
	#for all divisions call the corpus creator
	# parse command line parameters
	#Optional arguments:
	#-i string, path of source data (in swda style)
	#-t string, target path of folder for the preprocessed data
	#-f string, path of file with the division of files to be turned into
	#a corpus
	#-a string, path to disfluency annotations
	#-lm string, Location of where to write a clean language\
	#model files out of this corpus
	#-pos boolean, Whether to write a word2pos mapping folder
	#in the sister directory to the corpusLocation, else assume it is there
	#-p boolean, whether to include partial words or not
	#-d boolean, include dialogue act tags in the info
	print "Creating corpus..."
	write_pos_map = True
	for div in file_divisions_transcripts:
		command = [
		sys.executable, 
		'corpus/disfluency_corpus_creator.py',
		'-i', "data/raw_data/swda",
		'-t', "data/disfluency_detection/switchboard",
		'-f', div,
		'-a', "data/disfluency_detection/swda_disfluency_annotations",
		# '-lm', "data/lm_corpora",
		'-p',
		'-d'
		]
		if write_pos_map:
			command.append("-pos")
			write_pos_map = False #just call it once
		subprocess.call(command)
	print "Finished creating corpus."

#2. Run the preprocessing and extraction of features for all files, 
if extract_features:
	#note to get the audio feature extraction to work you need to have 
	#optional arguments are:
	#-i string, path of source disfluency corpus
	#-t string, target path of folder feature vectors in this folder
	# (rather than use text files)
	#-f string, path of file with the division of files to be turned into
	#a corpus of vectors
	#-p boolean, whether to include partial words or not
	#-a string, path to word alignment folder
	#-tag string, path of folder with tag representations
	#-new_tag bool, whether to write new tag representations or use old ones
	#-pos path, path to POS tagger if using one, if None use gold
	#-train_pos bool, whether to train pos tagger or not and put it in pos
	#-u bool, include utterance segmentation tags, derivable from utts
	#-d bool, include dialogue act tags
	#-l bool, include laughter tags on words- either speech laugh on word or 
	#bout
	#-joint bool, include big joint tag set as well as the individual ones
	#-lm string, Location of where to write a clean language\
	#model files out of this corpus
	#-xlm boolean, Whether to use a cross language model\
	#training to be used for getting lm features on the same data.
	#-asr boolean, whether to produce ASR results for creation of the 
	#data or not
	#-credentials string, username:password for IBM ASR
	#-audio string, path to open smile for audio features, if None
	#no audio extraction
	print "Extracting features..."
	tags_created = False
	tagger_trained = False
	for div in file_divisions_transcripts:
		command = [
			sys.executable,
			'feature_extraction/extract_features.py',
			'-i', "data/disfluency_detection/switchboard",
			'-t', "data/disfluency_detection/vectors",
			'-f', div,
			'-p',
			'-a', 'data/raw_data/swbd_alignments/alignments',
			'-tag', "data/tag_representations",
			'-u',
			'-d',
			'-l',
			'-joint',
			# '-lm', "data/lm_corpora"
			]
		if "train" in div and "-lm" in command:
			command.append("-xlm")
		if not tags_created:
			command.append("-new_tag")
			tags_created = True
		if asr and "ASR" in div:
			command.extend(["-pos","data/crfpostagger"])
			if not tagger_trained:
				command.append("-train_pos")
			credentials = "1841487c-30f4-4450-90bd-38d1271df295:EcqA8yIP7HBZ"
			command.extend(['-asr','-credentials',credentials])
		subprocess.call(command)
	print "Finished extracting features."
	

#3. Train the model on the transcripts (and audio data if available)
# NB each of these experiments can take up to 24 hours
if train_models:
	#until convergence
	#on the settings according to the numbered experiments in 
	#experiments/config.csv file
	for exp in experiments:
		command = [sys.executable,
					'experiments/run_experiment.py',
					'-exp', "0{0}".format(exp)]
	subprocess.call(command)
		
#4. Test the models on the test transcripts according to the best epochs
# from training.
# The output from the models is made in the folders
# For now all use timing data
if test_models:
	allsystemsfinal = [
					   (33,45,'RNN'),
					   (34,37,'RNN (complex tags)'),
					   (35,6,'LSTM'),
					   (36,15,'LSTM (complex tags)'),
					   (37,6,'LSTM (disf only)'),
                       (38,8,'LSTM (TTO only)'),
                       (39,2,'LSTM (complex tags)')
                       ]
	
	for exp, e, system in allsystemsfinal:
		print exp, e, system

		command = [sys.executable,
						'experiments/run_experiment.py',
						'-exp', "0{0}".format(exp),
						'-hmm',
						'-epoch', str(e)]
		subprocess.call(command)
		
#5. To get the numbers run the notebook:
#experiments/analysis/EACL_2017/EACL_2017.ipynb
# The results should be consistent with that in the EACL 2017 paper.
