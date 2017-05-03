from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger


def test_tagger():
    # Initialize the tagger from the config file with a config number
    # and saved model directory
    # 35 is the simple tagging model which gives utterance boundaries t/>
    # edit terms <e/>, interregna <i/> and repair onsets <rps
    disf = DeepDisfluencyTagger(
            config_file="deep_disfluency/experiments/experiment_configs.csv",
            config_number=35,
            saved_model_dir="deep_disfluency/experiments/035/epoch_6"
            )
    # Tag each word incrementally
    # Notice the incremental change in previous utterances
    # Set diff_only to False if you want the whole utterance's tag each time
    print disf.tag_new_word("john", pos="NNP", timing=0.33, diff_only=True)
    print disf.tag_new_word("likes", pos="VBP", timing=0.33, diff_only=True)
    print disf.tag_new_word("uh", pos="UH", timing=0.33, diff_only=True)
    print disf.tag_new_word("loves", pos="VBP", timing=0.33, diff_only=True)
    print disf.tag_new_word("mary", pos="NNP", timing=0.33, diff_only=True)
    print disf.tag_new_word("yeah", pos="UH", timing=2.00, diff_only=True)
    disf.reset()  # resets the whole tagger

    print "*" * 30
    # A more complex model with the whole repair structure
    disf = DeepDisfluencyTagger(
        config_file="deep_disfluency/experiments/experiment_configs.csv",
        config_number=36,
        saved_model_dir="deep_disfluency/experiments/036/epoch_15"
        )
    # Tag each word incrementally
    # Notice the incremental change in previous utterances
    # Set diff_only to False if you want the whole utterance's tag each time
    print disf.tag_new_word("i", pos="PRP", timing=0.33, diff_only=True)
    print disf.tag_new_word("uh", pos="UH", timing=0.33, diff_only=True)
    print disf.tag_new_word("i", pos="PRP", timing=0.33, diff_only=True)
    print disf.tag_new_word("love", pos="VBP", timing=0.33, diff_only=True)
    print disf.tag_new_word("mary", pos="NNP", timing=0.33, diff_only=True)
    print disf.tag_new_word("yeah", pos="UH", timing=2.00, diff_only=True)
    disf.reset()  # resets the whole tagger


if __name__ == '__main__':
    test_tagger()
