import logging
import os
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


# sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
switchboard_clean_directory = "../data/lm_corpora/"


class MySentences(object):
    # a memory-friendly iterator
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if "edit" in fname or 'heldout' in fname:
                continue  # no edit terms
            print fname
            for line in open(os.path.join(self.dirname, fname)):
                if "clean" in fname:
                    ID, text = line.split(",")
                else:
                    ID = ""
                    text = line
                if ID == "POS":
                    continue
                yield text.split()

sentences = MySentences(switchboard_clean_directory)
model = gensim.models.Word2Vec(sentences, min_count=2, size=50)
model.save('bnc_swbd_clean_50')
print model.index2word[0]
print len(model)
# print model['um']

# for i in model.index2word:
#     print i
