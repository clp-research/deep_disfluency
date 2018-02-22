# Deep Learning Driven Incremental Disfluency Detection

Code for Deep Learning driven incremental disfluency detection and related dialogue processing tasks.

## Set up and basic use ##

To run the code here you need to have `Python 2.7` installed, and also [`pip`](https://pip.readthedocs.org/en/1.1/installing.html) for installing the dependencies.

You need to run the below from the command line from inside this folder (depending on your user status, you may need to prefix the below with `sudo` or use a virtual environment):

`pip install -r requirements.txt`

If you just want to use the tagger off-the-shelf see the usage in `demo.py` or the notebook `demo.ipynb`.
Make sure this repository is on your system path if you want to use it in python more generally.

## Running experiments ##

The code can be used to run the experiments on Recurrent Neural Networks (RNNs) and LSTMs from:

```
Julian Hough and David Schlangen. Joint, Incremental Disfluency Detection and Utterance Segmentation from Speech. Proceedings of EACL 2017. Valencia, Spain, April 2017.
```

Please cite the paper if you use this code.

If you are using our pretrained models as in the usage in `demo.py` you can simply run `deep_disfluency/experiments/EACL_2017.py`, ensuring the boolean variables at the top of the file to:

```python
download_raw_data = False
create_disf_corpus = False
extract_features = False
train_models = False
test_models = True
```

If that level of reproducibility does not satisfy you, you can set all those boolean values to `True` (NB: be wary that training the models for each experiment in the script can take 24hrs+ even with a decent GPU).

Once the script has been run, running the Ipython notebook at `deep_disfluency/experiments/analysis/EACL_2017/EACL_2017.ipynb` should process the outputs and give similar results to those recorded in the paper.

*Acknowledgments*

This basis of these models is the disfluency and dialogue act annotated Switchboard corpus, based on that provided by Christopher Potts's 2011 Computational Pragmatics course ([[at http://compprag.christopherpotts.net/swda.html]]) or at [[https://github.com/cgpotts/swda]]. Here we use Julian Hough's fork which corrects some of the POS-tags and disfluency annotation:

[[https://github.com/julianhough/swda.git]]

The second basis is the word timings data for switchboard, which is a corrected version with word timing information to the Penn Treebank version of the MS alignments, which can be downloaded at:

[[http://www.isip.piconepress.com/projects/switchboard/releases/ptree_word_alignments.tar.gz]]

## Extra: using the Switchboard audio data ##

If you are satisfied just using lexical/POS/Dialogue Acts and word timing data alone, the above are sufficient, however if you want to use other acoustic data and use ASR results, you must have access to the Switchboard corpus audio release. This is available for purchase from:

[[https://catalog.ldc.upenn.edu/ldc97s62]]

From the switchboard audio release, copy or move the folder which contains the .sph files (called `swbd1`) to within the `deep_disfluency/data/raw_data/` folder. Note this is very large at around 14GB.

#Future: Creating your own data#

Training data is created through creating dialogue matrices (one per speaker in each dialogue), whereby the format of these for each row in the matrix is as follows, where `,` indicates a new column, and `...` means there are potentially multiple columns:

`word index, pos index, word duration, acoustic features..., lm features..., label`

There are methods for creating these in the `deep_disfluency/corpus` and `deep_disfluency/feature_extraction` modules.















