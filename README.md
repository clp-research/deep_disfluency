# Deep Learning Incremental Disfluency Detection

Deep Learning systems for training and testing disfluency detection and related tasks on speech data.

==============================================================================================================

Code for Deep Learning based Incremental Disfluency Detection. The code can be used to run the experiments on Recurrent Neural Networks in the Interspeech 2015 paper:

Julian Hough and David Schlangen. Recurrent Neural Networks for Incremental Disfluency Detection. INTERSPEECH 2015. Dresden, Germany, September 2015.

The code can also be used to run the experiments on Recurrent Neural Networks and LSTMs from:

Julian Hough and David Schlangen. Joint, Incremental Disfluency Detection and Utterance Segmentation from Speech. Proceedings of EACL 2017. Valencia, Spain, April 2017.

Please cite the appropriate paper if you use this code.

Acknowledgement: the code uses some of the core code from the Interspeech 2013 paper:

Gregoire Mesnil, Xiaodong He, Li Deng and Yoshua Bengio
Investigation of Recurrent Neural Network Architectures and
Learning Methods for Spoken Language Understanding

http://www.iro.umontreal.ca/~lisa/pointeurs/RNNSpokenLanguage2013.pdf

## Set up ##

To run the code here you need to have Python 2.7 installed, and also `pip` for installing the dependencies. (Also Ipython should be installed, preferrably, if you want to run notebooks).

Firstly, install Cython, then h5py, by running the below on the command line:

`sudo pip install Cython`

`sudo pip install h5py`

You then need to run the below from the command line from inside this folder:

`sudo pip install -r requirements.txt`

You then need to clone the Mumodo repository from https://github.com/dsg-bielefeld/mumodo.git and follow the installation instructions there.
