# Deep Learning Driven Incremental Disfluency Detection

Code for Deep Learning driven incremental disfluency detection and related dialogue processing tasks.

## Set up ##

To run the code here you need to have `Python 2.7` installed, and also `pip` for installing the dependencies. (Also `IPython` should be installed, preferrably, if you want to run notebooks).

Firstly, install Cython, then h5py, by running the below on the command line:

`sudo pip install Cython`

`sudo pip install h5py`

You then need to run the below from the command line from inside this folder:

`sudo pip install -r requirements.txt`

If you just want to use the tagger off-the-shelf see the usage in `demo.py` or the notebook `demo.ipynb`.
Make sure this repository is on your system path if you want to use it in python more generally.

## Running experiments ##

The code can be used to run the experiments on Recurrent Neural Networks in the Interspeech 2015 paper:

Julian Hough and David Schlangen. Recurrent Neural Networks for Incremental Disfluency Detection. INTERSPEECH 2015. Dresden, Germany, September 2015.

The code can also be used to run the experiments on Recurrent Neural Networks and LSTMs from:

Julian Hough and David Schlangen. Joint, Incremental Disfluency Detection and Utterance Segmentation from Speech. Proceedings of EACL 2017. Valencia, Spain, April 2017.

Please cite the appropriate paper if you use this code.

If you are using a pretrained model as in the usage in `demo.py` you can run `experiments/EACL_2017.py` or `experiments/Interspeech_2015.py` but adjusting the parameters at the top of the file to:

```python
create_disf_corpus = False
extract_features = False
train_models = False
test_models = True
```
 
If that level of reproducibility does not satisfy you, you can set all those boolean values to `True`. The below set-up of the data must then be done before running the appropriate script.

**Data**

Training is done through creating dialogue matrices (one per speaker in each dialogue), whereby the format of these for each row in the matrix is:

`word_idx, pos_idx, word_duration, acoustic_features..., lm_features...., label`


To generate the data for these experiments, for using text alone (without speech data) you need access to two publicly available versions of the Switchboard corpus transcripts.

The first is the disfluency and dialogue act annotated Switchboard corpus, based on that provided by Christopher Potts's 2011 Computational Pragmatics course (at http://compprag.christopherpotts.net/swda.html) or at https://github.com/cgpotts/swda. Clone Julian Hough's fork:

https://github.com/julianhough/swda.git

Extract the swda.zip folder from that repo as into the data/raw_data/ folder.

The second is to get the word timings, which is a corrected version with word timing information to the Penn Treebank version of the MS alignments, which can be downloaded at:

http://www.isip.piconepress.com/projects/switchboard/releases/ptree_word_alignments.tar.gz

Extract the folder into the data/raw_data/ directory and rename the root of the extracted folder from 'data' to 'swbd_alignments'. 

If you are satisfied just using lexical/POS/Dialogue Acts and word timing data alone, the above are sufficient, however if you want to use other acoustic data and use ASR results, you must have access to the Switchboard corpus audio release. This is available for purchase from:

https://catalog.ldc.upenn.edu/ldc97s62

From the switchboard audio release, copy or move the folder which contains the .sph files (swbd1) to within the data/raw_data folder. Note this is very large at around 14GB.

By now the raw_data folder will have the following structure (with the possible omission of the swbd1 folder with the audio data if that is not required):

```bash
deep_disfluency$ tree --filelimit 15 deep_disf/data/raw_data/
deep_disf/data/raw_data/
├── README.txt
├── swbd_alignments
│   ├── AAREADME.text
│   ├── alignments
│   │   ├── 2 [910 entries exceeds filelimit, not opening dir]
│   │   ├── 3 [954 entries exceeds filelimit, not opening dir]
│   │   └── 4 [388 entries exceeds filelimit, not opening dir]
│   └── manual.text
└── swda
    ├── sw00utt [99 entries exceeds filelimit, not opening dir]
    ├── sw01utt [100 entries exceeds filelimit, not opening dir]
    ├── sw02utt [100 entries exceeds filelimit, not opening dir]
    ├── sw03utt [100 entries exceeds filelimit, not opening dir]
    ├── sw04utt [100 entries exceeds filelimit, not opening dir]
    ├── sw05utt [100 entries exceeds filelimit, not opening dir]
    ├── sw06utt [100 entries exceeds filelimit, not opening dir]
    ├── sw07utt [100 entries exceeds filelimit, not opening dir]
    ├── sw08utt [100 entries exceeds filelimit, not opening dir]
    ├── sw09utt [100 entries exceeds filelimit, not opening dir]
    ├── sw10utt [100 entries exceeds filelimit, not opening dir]
    ├── sw11utt [16 entries exceeds filelimit, not opening dir]
    ├── sw12utt
    │   ├── sw_1200_2121.utt.csv
    │   ├── sw_1201_2131.utt.csv
    │   ├── sw_1202_2151.utt.csv
    │   ├── sw_1203_2229.utt.csv
    │   ├── sw_1204_2434.utt.csv
    │   ├── sw_1205_2441.utt.csv
    │   ├── sw_1206_2461.utt.csv
    │   ├── sw_1207_2503.utt.csv
    │   ├── sw_1208_2724.utt.csv
    │   ├── sw_1209_2836.utt.csv
    │   └── sw_1210_3756.utt.csv
    ├── sw13utt [29 entries exceeds filelimit, not opening dir]
    └── swda-metadata.csv
```

If the above is in place (even without acoustic features), the script at `deep_disfluency/experiments/EACL_2017.py` should work out of the box and return the results from the paper.














