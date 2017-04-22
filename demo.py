from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger


def test_tagger():
    # Initialize the tagger from the config file with a config number
    disf = DeepDisfluencyTagger(
            config_file="deep_disfluency/experiments/experiment_configs.csv",
            config_number=35,
            saved_model_dir="deep_disfluency/experiments/035/epoch_6"
            # optional
            )
    # Tag each word individually and it outputs the whole tag sequence
    # each time
    # Notice the incremental change in previous utterances
    disf.tag_new_word("john", pos="NNP", timing=0.33)
    disf.tag_new_word("likes", pos="VBP", timing=0.33)
    disf.tag_new_word("uh", pos="UH", timing=0.33)
    disf.tag_new_word("loves", pos="VBP", timing=0.33)
    disf.tag_new_word("mary", pos="NNP", timing=0.33)
    disf.tag_new_word("yeah", pos="UH", timing=2.00)
    disf.reset()  # resets the whole tagger

    # if you want to train a net from data
    # feed a set of dialogue vectors to the model for training
#     dialogues = None
#     dir_path = ""
#     decoder_path = ""
#     disf.train_net(dialogues)
#     # you can override the arguments that would be specified
#     # in the config file if you want
#     disf.args.n_epochs = 10
#     disf.save_net_weights_to_dir(dir_path)
#     # you can train the the decoder
#     disf.train_decoder(dialogues)
#     disf.save_decoder_model(decoder_path)

if __name__ == '__main__':
    test_tagger()
