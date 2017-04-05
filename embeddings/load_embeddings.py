import numpy


def populate_embeddings(emb_dim, vocsize, words2index, pretrained):
    """From pre-trained embeddings and matrix dimension generates a random
     matrix first,
    then assings the appropriate slices from the pre-learned embeddings"""
    assert pretrained.layer1_size == emb_dim, str(pretrained.layer1_size) + \
        " " + str(emb_dim)
    emb = 0.2 * numpy.random.uniform(-1.0, 1.0,
                                     (vocsize+1, emb_dim)).astype('Float32')
    for i in range(0, len(pretrained.index2word)):
        word = pretrained.index2word[i]
        # print i, word # returns the word and its index in pretrained
        emb[words2index[word]] = pretrained[word]  # assign the correct index
    return emb
    # the embeddings without a corresponding word inherit random init.
