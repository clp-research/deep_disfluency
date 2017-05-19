from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger


# Initialize the tagger from the config file with a config number
# and saved model directory
print "1. Disfluency tagging on pre-segmented utterances"
print "tags repair structure incrementally and other edit terms <e/>"
print "(Hough and Schlangen Interspeech 2015 with an RNN)"
disf = DeepDisfluencyTagger(
    config_file="deep_disfluency/experiments/experiment_configs.csv",
    config_number=21,
    saved_model_dir="deep_disfluency/experiments/021/epoch_40"
    )

# Tag each word incrementally
# Notice the incremental diff
# Set diff_only to False if you want the whole utterance's tag each time
print "tagging..."
print disf.tag_new_word("john", pos="NNP", diff_only=True)
print disf.tag_new_word("likes", pos="VBP", diff_only=True)
print disf.tag_new_word("uh", pos="UH", diff_only=True)
print disf.tag_new_word("loves", pos="VBP", diff_only=True)
print disf.tag_new_word("mary", pos="NNP", diff_only=True)
print "final tags:"
for w, t in zip("john likes uh loves mary".split(), disf.output_tags):
    print w, "\t", t
disf.reset()  # resets the whole tagger for new utterance

# More complex set-up:
print "\n", "*" * 30
print "2. Joint disfluency tagger and utterance semgenter"
print "Simple disf tags <e/>, <i/> and repair onsets <rps"
print "LSTM simple from Hough and Schlangen EACL 2017"
disf = DeepDisfluencyTagger(
        config_file="deep_disfluency/experiments/experiment_configs.csv",
        config_number=35,
        saved_model_dir="deep_disfluency/experiments/035/epoch_6",
        use_timing_data=True
        )


print "tagging..."
print disf.tag_new_word("john", pos="NNP", timing=0.33, diff_only=True)
print disf.tag_new_word("likes", pos="VBP", timing=0.33, diff_only=True)
print disf.tag_new_word("uh", pos="UH", timing=0.33, diff_only=True)
print disf.tag_new_word("loves", pos="VBP", timing=0.33, diff_only=True)
print disf.tag_new_word("mary", pos="NNP", timing=0.33, diff_only=True)
print disf.tag_new_word("yeah", pos="UH", timing=2.00, diff_only=True)
print "final tags:"
for w, t in zip("john likes uh loves mary yeah".split(), disf.output_tags):
    print w, "\t", t
disf.reset()  # resets the whole tagger for next dialogue or turn


print "\n", "*" * 30
print "3. Joint disfluency tagger and utterance semgenter"
print "Full complex tag set with disfluency structure"
print "LSTM complex from Hough and Schlangen EACL 2017"
disf = DeepDisfluencyTagger(
    config_file="deep_disfluency/experiments/experiment_configs.csv",
    config_number=36,
    saved_model_dir="deep_disfluency/experiments/036/epoch_15",
    use_timing_data=True
    )

print "tagging..."
print disf.tag_new_word("i", pos="PRP", timing=0.33, diff_only=True)
print disf.tag_new_word("uh", pos="UH", timing=0.33, diff_only=True)
print disf.tag_new_word("i", pos="PRP", timing=0.33, diff_only=True)
print disf.tag_new_word("love", pos="VBP", timing=0.33, diff_only=True)
print disf.tag_new_word("mary", pos="NNP", timing=0.33, diff_only=True)
print disf.tag_new_word("yeah", pos="UH", timing=2.00, diff_only=True)
print "final tags:"
for w, t in zip("i uh i love mary yeah".split(), disf.output_tags):
    print w, "\t", t
disf.reset()  # resets the whole tagger
