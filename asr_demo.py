from deep_disfluency.asr.ibm_watson import IBMWatsonASR
from copy import deepcopy
from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger

disf = DeepDisfluencyTagger(
        config_file="deep_disfluency/experiments/experiment_configs.csv",
        config_number=35,
        saved_model_dir="deep_disfluency/experiments/035/epoch_6",
        use_timing_data=True
       )


# define what you want to happen when new word hypotheses come in
def new_word_hypotheses_handler(word_diff, rollback, word_graph):
    # print "NEW HYPOTHESIS:%%%%%%"
    # print word_diff
    # roll back the disf tagger
    if word_diff == []:
        return
    disf.rollback(rollback)
    # print "current", disf.word_graph
    last_end_time = 0
    if len(word_graph) > 1:
        last_end_time = word_graph[(len(word_graph)-len(word_diff))-1][-1]
    # tag new words and work out where the new tuples start
    new_output_start = max([0, len(disf.get_output_tags())-1])
    for word, _, end_time in word_diff:
        timing = end_time - last_end_time
        new_tags = disf.tag_new_word(word, timing=timing)
        start_test = max([0, len(disf.get_output_tags())-len(new_tags)-1])
        if start_test < new_output_start:
            new_output_start = start_test
        # print "new tag diff:"
        # print new_tags
        last_end_time = end_time
    # print out all the output tags from the new word diff onwards
    print "\nnew tags:"
    for h in disf.get_output_tags(with_words=True)[new_output_start:]:
        print h

# NB! Put your Watson credentials as credentials.json in this directory
asr = IBMWatsonASR("credentials.json", new_word_hypotheses_handler)
asr.listen()
