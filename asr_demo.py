from deep_disfluency.asr.ibm_watson import IBMWatsonASR
from copy import deepcopy
from deep_disfluency.tagger.deep_tagger import DeepDisfluencyTagger
import subprocess
from itertools import count
from tkinter import Tk, Label, Button


class DisfluencyGUI:

    def __init__(self, master, disf_tagger):

        # NB! Put your Watson credentials as credentials.json in this directory
        self.disf = disf_tagger
        self.master = master
        self.master.title("Disfluency Counter")

        # self.label = Label(master, text="Disfluency Counter")
        # self.label.pack()

        self.label = Label(self.master, text="0", fg="red")
        self.label.pack()

        self.close_button = Button(self.master,
                                   text="Close", command=self.master.quit)
        self.close_button.pack()
        self.counter = count(0)
        self.words_with_disfluency = []

        def new_word_hypotheses_handler(word_diff, rollback, word_graph):
            # print "NEW HYPOTHESIS:%%%%%%"
            # print word_diff
            # roll back the disf tagger
            try:
                if word_diff == []:
                    return
                disf.rollback(rollback)
                # print "current", disf.word_graph
                last_end_time = 0
                if len(word_graph) > 1:
                    word_idx = (len(word_graph) - len(word_diff))-1
                    last_end_time = word_graph[word_idx][-1]
                # tag new words and work out where the new tuples start
                new_output_start = max([0, len(disf.get_output_tags())-1])
                for word, _, end_time in word_diff:
                    timing = end_time - last_end_time
                    new_tags = self.disf.tag_new_word(word, timing=timing)
                    end_idx = len(self.disf.get_output_tags())-len(new_tags)-1
                    start_test = max([0, end_idx])
                    if start_test < new_output_start:
                        new_output_start = start_test
                    # print "new tag diff:"
                    # print new_tags
                    last_end_time = end_time
                # print out all the output tags from the new word diff onwards

                print "\nnew tags:"
                for w, h, i in zip(
                    word_graph[
                        new_output_start:],
                    self.disf.get_output_tags(with_words=False)[
                        new_output_start:],
                    range(
                        new_output_start, len(word_graph))
                        ):
                    print w, h, i
                    if ("<e" in h or "<rps" in h) and \
                            i not in self.words_with_disfluency:
                        self.label.config(text=str(self.counter.next()))
                        self.master.update()
                        self.words_with_disfluency.append(i)
            except:
                print "FAILED TO UPDATE"

        self.master.update()
        self.asr = IBMWatsonASR("credentials.json",
                                new_word_hypotheses_handler)
        self.asr.listen()

if __name__ == '__main__':
    disf = DeepDisfluencyTagger(
        config_file="deep_disfluency/experiments/experiment_configs.csv",
        config_number=35,
        saved_model_dir="deep_disfluency/experiments/035/epoch_6",
        use_timing_data=True
    )
    root = Tk()
    my_gui = DisfluencyGUI(root, disf)
