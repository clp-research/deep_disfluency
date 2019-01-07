# corpus module

This module can stand alone apart from the tagging code, but included
here for convenience/use by other modules.

The main purpose is to convert from several different sources into a 
unified tagging format, which can include timing information optional.

Supported conversion so far:

- swda- Chris Potts's version of the Switchboard dialogue act corpus
and the Mississippi word timings specifically,
though with effort, other versions of Switchboard
- DUEL project data (Hough et al, 2016, LREC), converting from textgrids, 
though any textgrids with the DUEL-style annotations could work with effort.


## For using textgrid conversion, run "sudo pip install -r requirements.txt" from the command line from this folder.

You then need to clone the Mumodo repository from https://github.com/dsg-bielefeld/mumodo.git and follow the installation instructions there.
