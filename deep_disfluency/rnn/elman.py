import theano
import numpy as np
import os
import sys
import time

from theano import tensor as T
from collections import OrderedDict
from deep_disfluency.load.load import load_data_from_array

# nb might be theano.config.floatX
dtype = T.config.floatX  # @UndefinedVariable

class Elman(object):
    
    def __init__(self, ne, de, na, nh, n_out, cs, npos, update_embeddings=True):
        '''
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        na :: number of acoustic or language model features at each word step (acoustic context size in frames * number of features)
        nh :: dimension of the hidden layer
        n_out :: number of classes
        cs :: word window context size 
        npos :: number of pos tags
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (ne+1, de)).astype(dtype)) # add one for PADDING at the end
        #=======================================================================
        # self.Wx  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
        #            (de * cs, nh)).astype(theano.config.floatX))
        #=======================================================================
        
        if na==0:
            #NB original one, now Wx becomes much bigger with acoustic data
            self.Wx  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                    ((de * cs)+(npos * cs), nh)).astype(dtype))
        else:    
            self.Wx  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                    ((de * cs)+(npos * cs)+na, nh)).astype(dtype))
        self.Wh  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(dtype))
        self.W   = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (nh, n_out)).astype(dtype))
        self.bh  = theano.shared(np.zeros(nh, dtype=dtype))
        self.b   = theano.shared(np.zeros(n_out, dtype=dtype))
        self.h0  = theano.shared(np.zeros(nh, dtype=dtype))
        #TODO bit of a hack, just using the eye function (diagonal 1s) for the POS, small in memory
        self.pos = T.eye(npos,npos,0)
        
        self.n_acoust = na #the number of acoustic features
        
        
        #Weights for L1 and L2
        self.L1_reg = 0.0
        self.L2_reg = 0.00001


        # bundle
        self.params = [ self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ] #without embeddings updates
        self.names  = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        if update_embeddings:
            self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
            self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        
        self.idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
    
        self.pos_idxs = T.imatrix()
        
        #TODO new, simply a matrix that will the size of the incoming acoustic data
        #will come 'pre-flattened' unlike the emebdding input to the hidden layer
        self.acoustic = T.matrix() #the acoustic features associated with each word window
        
        #TODO Old version no pos
        #x = self.emb[self.idxs].reshape((self.idxs.shape[0], de*cs))
        
        if na == 0:
            #TODO POS version, not just the embeddings, but with the POS window concatenated?
            x = T.concatenate((self.emb[self.idxs].reshape((self.idxs.shape[0], de*cs)),\
                               self.pos[self.pos_idxs].reshape((self.pos_idxs.shape[0],npos*cs))), 1)
        else:
            #TODO new version with acoustic data
            x = T.concatenate((self.emb[self.idxs].reshape((self.idxs.shape[0], de*cs)),\
                               self.pos[self.pos_idxs].reshape((self.pos_idxs.shape[0],npos*cs)),\
                               self.acoustic), 1)
        self.y = T.iscalar('y') # label
        #TODO for sentences
        #self.y = T.ivector('y') #labels for whole sentence
        
        
        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        p_y_given_x_sentence_hidden = (h,s[:,0,:])
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        
        #TODO adding this- zero one loss for the last word
        y_pred_word = T.argmax(p_y_given_x_lastword)

        #learning rate not hard coded as could decay
        self.lr = T.scalar('lr')
        
        #Cost: standard nll loss
        self.nll = -T.mean(T.log(p_y_given_x_lastword)[self.y])
        
        self.sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), self.y])
        
        
        if na == 0:
            self.classify = theano.function(inputs=[self.idxs,self.pos_idxs], outputs=y_pred)
        else:
            self.classify = theano.function(inputs=[self.idxs,self.pos_idxs, self.acoustic], outputs=y_pred)
        
        #regularisation terms
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        # if not using this set this to 0 to avoid unecessary computation
        self.L1 = 0
        #self.L1 = abs(self.Wh.sum()) +  abs(self.Wx.sum()) +  abs(self.W.sum()) +  abs(self.emb.sum())\
        #            +  abs(self.bh.sum()) + abs(self.b.sum()) + abs(self.h0.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr =  (self.Wh ** 2).sum() + (self.Wx ** 2).sum() + (self.W ** 2).sum() +  (self.emb ** 2).sum()\
                        +  (self.bh ** 2).sum() +  (self.b ** 2).sum() + (self.h0 ** 2).sum()
        
        
        
        
        #the gradients to compute
        #gradients = T.grad( self.nll, self.params )
        #TODO am changing to below
        
        # costs for single label per input sequence
        self.cost = self.nll \
            + self.L1_reg * self.L1 \
            + self.L2_reg * self.L2_sqr
        gradients = T.grad( self.cost, self.params )
        
        self.updates = OrderedDict(( p, p-self.lr*g ) for p, g in zip( self.params , gradients))
        
        #costs for multiple labels (one for each in the input)
        self.sentence_cost = self.sentence_nll \
            + self.L1_reg * self.L1 \
            + self.L2_reg * self.L2_sqr
        sentence_gradients = T.grad(self.sentence_cost, self.params)
        
        self.sentence_updates = OrderedDict((p, p - self.lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))
        
        
        # theano functions #NB added POS
        #self.soft_max = theano.function(inputs=[self.idxs, self.pos_idxs], outputs=p_y_given_x_sentence) #simply outputs the soft_max distribution for each word in utterance
        #TODO added acoustic
        if na == 0:
            self.soft_max = theano.function(inputs=[self.idxs, self.pos_idxs],
                                            outputs=p_y_given_x_sentence)
            self.soft_max_return_hidden_layer = theano.function(
                                    inputs=[self.idxs, self.pos_idxs],
                                    outputs=p_y_given_x_sentence_hidden)
        else:
            self.soft_max = theano.function(inputs=[self.idxs, self.pos_idxs,
                                                    self.acoustic],
                                            outputs=p_y_given_x_sentence)
            self.soft_max_return_hidden_layer = theano.function(
                                        inputs=[self.idxs, self.pos_idxs,
                                                self.acoustic],
                                        outputs=p_y_given_x_sentence_hidden)

        #=======================================================================
        # #ORIGINAL CODE
        # self.train = theano.function( inputs  = [self.idxs, self.y, lr],
        #                               outputs = nll,
        #                               updates = self.updates )
        #=======================================================================
        if na == 0:
            self.train = theano.function(inputs=[self.idxs, self.pos_idxs,
                                                 self.y, self.lr],
                                         outputs=self.nll,
                                         updates=self.updates)
        else:
            self.train = theano.function(inputs=[self.idxs, self.pos_idxs,
                                                 self.acoustic, self.y,
                                                 self.lr],
                                         outputs=self.nll,
                                         updates=self.updates)
        
        
        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})
        
        #TODO new training method which uses acoustic/other real valued input
        #self.batchstart = T.iscalar('batchstart')
        #self.batchstop = T.iscalar('batchstop')
        
        #we precompile the data variables as shared from file, though they may change when training
        #self.lexical_data = self.shared_dataset(np.zeros((de*cs,1), dtype=theano.config.floatX)) #loads dummy data set as a shared variable
        #self.pos_data = self.shared_dataset(np.zeros((npos*cs,1), dtype=theano.config.floatX))
        #self.acoustic_data = self.shared_dataset(np.zeros((na,1), dtype=theano.config.floatX))
        #self.labels = self.shared_dataset((1,1))
        
        #=======================================================================
        # self.lexical_data = T.imatrix('lexicaldata') #loads dummy data set as a shared variable
        # self.pos_data = T.imatrix('posdata')
        # self.acoustic_data = T.matrix('acousticdata')
        # self.labels = T.ivector('labels')
        # 
        # #function to return cost
        # self.train_by_index_acoustic = theano.function( inputs  = [self.batchstart, self.batchstop, self.lr],
        #                                                         outputs = self.cost,
        #                                                         updates = self.updates, 
        #                                                         givens={self.idxs : self.lexical_data[self.batchstart:self.batchstop+1,:],
        #                                                                 self.pos_idxs : self.pos_data[self.batchstart:self.batchstop+1,:],
        #                                                                 self.acoustic : self.acoustic_data[self.batchstop,:],
        #                                                                 self.y : self.labels[self.batchstop]},
        #                                                        on_unused_input='warn')  
        #=======================================================================


    def fit(self, dialogue):
        """Fit method that takes pickled numpy matrices pathed to in the list
        files as its input.
        """
        raise NotImplementedError
#         print "training"
#         print "acoustic", acoustic
#         print "load data", load_data
#         loss = 0.0 #will increment the loss as we go along
#         tic = time.time()
#         current_index = 0
#         for i in range(0,len(dialogues)):
#             data = dialogues[i][1]
#             if load_data: #set to False on Bender for training as these will be python np objects there is enough space?
#                 data = np.load(data) #load the pickled numpy array
#                 _, acoustic_data, lex_data, pos_data, indices, labels = load_data_from_array(data, self.n_acoust)
#             else:
#                 _, acoustic_data, lex_data, pos_data, indices, labels = data #should be bundles up already
#             nw = acoustic_data.shape[0] # number of examples in this dialogue
#             #if acoustic:
#             #shuffle([train_lex, train_y], s['seed'])
#             #tic = time.time()
#             #mycorpus, myb_indices = corpusToIndexedMatrix(lex_data, , s['bs']) #window size across number of words deep, gets matrix too
#             #mylabels = np.asarray(list(itertools.chain(*train_y)), dtype='int32')
#             
#             
#             test = 0
#             #load in the data to shared vars, can use 'set value too'
#             #self.lexical_data = self.shared_dataset(lex_data) #loads dummy data set as a shared variable
#             #self.pos_data = self.shared_dataset(pos_data)
#             #self.acoustic_data = data = self.shared_dataset(acoustic_data,dtype=theano.config.floatX)
#             #self.labels = self.shared_dataset(labels)
#             for start,stop in indices:
#                 current_index+=1
#                 test+=1 #TODO for testing
#                 #if test > 50: break #TODO for testing
#                 if acoustic:
#                     #print 'acoustic raw', acoustic_data[start:stop+1,:].shape
#                     #ac = np.asarray(acoustic_data[start:stop+1,:],dtype='float32')
#                     #print 'acoustic', ac.shape
#                     #print 'lexical', lex_data[start:stop+1,:].shape
#                     #raw_input()
#                     x = self.train(lex_data[start:stop+1,:],pos_data[start:stop+1,:],np.asarray(acoustic_data[start:stop+1,:],dtype='float32'),labels[stop],lr)
#                 else:
#                     #print lex_data[start:stop+1,:]
#                     #print pos_data[start:stop+1,:]
#                     x = self.train(lex_data[start:stop+1,:],pos_data[start:stop+1,:],labels[stop],lr)
#                     #raw_input()
#                     
#                 loss+=x
#                 self.normalize()
#                 
#                 #print '[learning] >> %2.2f%%'%((stop+1)*100./nw),'of file {} / {}'.format(i+1,len(dialogues)),\
#             print 'file {} / {}'.format(i+1,len(dialogues)),'completed in %.2f (sec) <<\r'%(time.time()-tic)
#             sys.stdout.flush()
#             print "current train_loss", loss/float(current_index)
#             #break #TODO switch back
#         return loss/float(current_index)
                        
                
    def fit_old(self, my_seq, my_indices, my_labels, lr, nw, pos=None, sentence=False):
        """The main training function over the corpus indexing into my_indices"""
        #TODO need to compile all theano functions at initial rnn creation
        tic = time.time()
        corpus = self.shared_dataset(my_seq) #loads data set as a shared variable
        labels = self.shared_dataset(my_labels)
        if pos:
            pos = self.shared_dataset(pos)
        #TODO new effort to index the shared vars
        batchstart = T.iscalar('batchstart')
        batchstop = T.iscalar('batchstop')
        acoustic = T.imatrix()
        if sentence:
            cost = self.sentence_nll
            updates = self.sentence_updates
        else:    
            cost = self.nll
            updates = self.updates
        if pos is None:
            self.train_by_index = theano.function( inputs  = [batchstart, batchstop,self.lr],
                                      outputs = cost,
                                      updates = updates, 
                                      givens={self.idxs : corpus[batchstart:batchstop+1],
                                              self.y : labels[batchstop]},
                                      on_unused_input='warn')  
            #TODO have changed  self.y : labels[batchstop]}, 
        else:
            self.train_by_index = theano.function( inputs  = [batchstart, batchstop,self.lr],
                                      outputs = cost,
                                      updates = updates, 
                                      givens={self.idxs : corpus[batchstart:batchstop+1],
                                              self.pos_idxs : pos[batchstart:batchstop+1],
                                              self.y : labels[batchstop]},
                                      on_unused_input='warn') 
            #TODO getting acoustic info in
            self.train_by_index = theano.function( inputs  = [batchstart, batchstop, acoustic, self.lr],
                                      outputs = cost,
                                      updates = updates, 
                                      givens={self.idxs : corpus[batchstart:batchstop+1],
                                              self.pos_idxs : pos[batchstart:batchstop+1],
                                              self.acoustic : acoustic,
                                              self.y : labels[batchstop]},
                                      on_unused_input='warn') 
            
            #TODO have changed  self.y : labels[batchstop]},   
        train_loss = 0.0
        laststop = 0
        i= 0
        for start,stop in my_indices:
            laststop = stop
            i+=1 #TODO for testing
            if i> 50: break
            x = self.train_by_index(start,stop,lr)
            train_loss+=x
            #print i
            self.normalize()
            #if stop % 6500 == 0 and stop>0:
            print '[learning] >> %2.2f%%'%((stop+1)*100./nw),'completed in %.2f (sec) <<\r'%(time.time()-tic),
            sys.stdout.flush()
        print "train_loss (may include reg)"
        print train_loss/float(laststop)
        return train_loss/float(laststop)
                
    def corpus_nll_old(self, my_seq, my_indices, my_labels, pos=None, sentence=False):
        """Computes the average loss (per word) for whole corpus"""
        # batch sizes is a list the same length as losses
        # which has size of each batch in it (all of sizefrom 1 to bptt limit). Computes with myindices
        #batch_sizes = [(my_indices[i][1]-my_indices[i][0])+1 for i in range(0,len(my_indices))] 
        #print len(batch_sizes) #TODO don't actually need this for weighting the loss as this done anyway I believe by length of context sequence?

        corpus = self.shared_dataset(my_seq) #loads data set as a shared variable
        labels = self.shared_dataset(my_labels)
        if pos is not None:
            pos = self.shared_dataset(pos)
        
        batchstart = T.iscalar('batchstart')
        batchstop = T.iscalar('batchstop')
        if sentence == True:
            cost = self.sentence_nll
        else:    
            cost = self.nll
        
        if pos is None:
            self.error_by_index = theano.function( inputs  = [batchstart, batchstop],
                                          outputs = cost,
                                          givens={self.idxs : corpus[batchstart:batchstop+1],
                                                  self.y : labels[batchstop]},
                                          on_unused_input='warn')
            #TODO have changed  self.y : labels[batchstop]},    
        
        else:
            self.error_by_index = theano.function( inputs  = [batchstart, batchstop],
                                          outputs = cost,
                                          givens={self.idxs : corpus[batchstart:batchstop+1],
                                                  self.pos_idxs : pos[batchstart:batchstop+1],
                                                  self.y : labels[batchstop]},
                                          on_unused_input='warn')   
            #TODO have changed  self.y : labels[batchstop]}, 
            
        losses = [ self.error_by_index(start,stop) for start,stop in my_indices ]

        return np.average(losses) 
    
     
    def shared_dataset(self, mycorpus, borrow=True, data_type='int32'):
        """ Load the dataset into shared variables """
        return theano.shared(np.asarray(mycorpus, dtype=data_type),
                                 borrow=True)
    
    def load_weights_from_folder(self, folder):
        for name, param in zip(self.names, self.params):
            param.set_value(np.load(os.path.join(folder, name + ".npy")))
            
    def load(self, folder):
        emb = np.load(os.path.join(folder, 'embeddings.npy'))
        Wx = np.load(os.path.join(folder, 'Wx.npy'))
        Wh = np.load(os.path.join(folder, 'Wh.npy'))
        W = np.load(os.path.join(folder, 'W.npy'))
        bh = np.load(os.path.join(folder, 'bh.npy'))
        b = np.load(os.path.join(folder, 'b.npy'))
        h0 = np.load(os.path.join(folder, 'h0.npy'))
        return emb, Wx, Wh, W, bh, b, h0

    def load_weights(self, emb=None, Wx=None, Wh=None, W=None, bh=None, b=None,
                     h0=None):
        if emb is not None:
            self.emb.set_value(emb)
        if Wx is not None:
            self.Wx.set_value(Wx)
        if Wh is not None:
            self.Wh.set_value(Wh)
        if W is not None:
            self.W.set_value(W)
        if bh is not None:
            self.bh.set_value(bh)
        if b is not None:
            self.b.set_value(b)
        if h0 is not None:
            self.h0.set_value(h0)
    
    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())