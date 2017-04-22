To extract the features for the experiments, you must run first make a sister folder for this repository and call it 'swbd'.

Within swbd/ you must have the subdirectories MSaligned, which is the Mississippi University swbd transcriptions with word timings and also a sub-directory switchboard_audio/. This is the untarred LDC release of switchboard with the .sph files.

When the folder structure is as above, then run 'extract_features.py'. You may uncomment out the execfile commands in this file if any of the steps are not needed.