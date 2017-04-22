import numpy as np
import os
import sys
import time
import theano
import theano.tensor as T
from theano import shared
from collections import OrderedDict

from deep_disfluency.load.load import load_data_from_array


def init_weight(shape, name, sample='uni', seed=None):
    rng = np.random.RandomState(seed)
    if sample == 'unishape':
        values = rng.uniform(
            low=-np.sqrt(6. / (shape[0] + shape[1])),
            high=np.sqrt(6. / (shape[0] + shape[1])),
            size=shape).astype(dtype)
    elif sample == 'svd':
        values = rng.uniform(low=-1., high=1., size=shape).astype(dtype)
        _, svs, _ = np.linalg.svd(values)
        # svs[0] is the largest singular value
        values = values / svs[0]
    elif sample == 'uni':
        values = rng.uniform(low=-0.1, high=0.1, size=shape).astype(dtype)
    elif sample == 'zero':
        values = np.zeros(shape=shape, dtype=dtype)
    else:
        raise ValueError("Unsupported initialization scheme: %s"
                         % sample)

    return shared(values, name=name, borrow=True)

dtype = T.config.floatX  # @UndefinedVariable
#print "loaded lstm.py"


class LSTM(object):

    def __init__(self, ne, de, na, n_lstm, n_out, cs, npos, lr=0.05, 
                 single_output=True, output_activation=T.nnet.softmax, 
                 cost_function='nll'):
        '''
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        na :: number of acoustic features at each word step
                (acoustic context size in frames * number of features)
        n_lstm :: dimension of the lstm layer
        n_out :: number of classes
        cs :: word window context size
        npos :: number of pos tags
        '''

        self.emb = init_weight((ne+1, de),'emb') # add one to ne for PADDING at the end
        #=======================================================================
        # self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
        #            (de * cs, nh)).astype(theano.config.floatX))
        #=======================================================================
        
        self.n_in = (de * cs)+(npos * cs)
        self.n_lstm = n_lstm
        self.n_out = n_out
        self.W_xi = init_weight((self.n_in, self.n_lstm),'W_xi') #input gate weights
        #self.W_xi  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
        #            ((de * cs)+(npos * cs), n_lstm)).astype(theano.config.floatX))
        self.W_hi = init_weight((self.n_lstm, self.n_lstm),'W_hi', 'svd') 
        self.W_ci = init_weight((self.n_lstm, self.n_lstm),'W_ci', 'svd') 
        self.b_i = shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_lstm))) #bias to the input
        self.W_xf = init_weight((self.n_in, self.n_lstm),'W_xf')  #forget gate weights
        self.W_hf = init_weight((self.n_lstm, self.n_lstm),'W_hf', 'svd') 
        self.W_cf = init_weight((self.n_lstm, self.n_lstm),'W_cf', 'svd') 
        self.b_f = shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_lstm))) #bias
        self.W_xc = init_weight((self.n_in, self.n_lstm),'W_xc') #memory cell gate weights
        self.W_hc = init_weight((self.n_lstm, self.n_lstm),'W_hc', 'svd') 
        self.b_c = shared(np.zeros(n_lstm, dtype=dtype)) #bias to the memory cell
        self.W_xo = init_weight((self.n_in, self.n_lstm),'W_xo') #output gate weights
        self.W_ho = init_weight((self.n_lstm, self.n_lstm),'W_ho', 'svd') 
        self.W_co = init_weight((self.n_lstm, self.n_lstm),'W_co', 'svd') 
        self.b_o = shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_lstm))) #bias on output gate
        self.W_hy = init_weight((self.n_lstm, self.n_out),'W_hy') #hidden to y weight
        self.b_y = shared(np.zeros(n_out, dtype=dtype)) #output bias
        
        
        #Weights for L1 and L2
        self.L1_reg = 0.0
        self.L2_reg = 0.00001
        
        
        
        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i, 
                       self.W_xf, self.W_hf, self.W_cf, self.b_f, 
                       self.W_xc, self.W_hc, self.b_c, 
                       self.W_ho, self.W_co, self.W_co, self.b_o, 
                       self.W_hy, self.b_y, self.emb]
        self.names  = ["W_xi", "W_hi", "W_ci", "b_i", 
                       "W_xf", "W_hf", "W_cf", "b_f", 
                       "W_xc", "W_hc", "b_c", 
                       "W_ho", "W_co", "W_co", "b_o", 
                       "W_hy", "b_y", "embeddings"]
                
        def step_lstm(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
            c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.b_c) 
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo)+ T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co)  + self.b_o)
            h_t = o_t * T.tanh(c_t)
            y_t = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.b_y) 
            return [h_t, c_t, y_t]

        self.idxs = T.imatrix() # batch of sequence of vectors
        self.pos_idxs = T.imatrix()
        
        #TODO bit of a hack, just using the eye function (diagonal 1s) for the POS, small in memory
        self.pos = T.eye(npos,npos,0)
        
        #TODO Old version no pos
        #x = self.emb[self.idxs].reshape((self.idxs.shape[0], de*cs))
        #TODO POS version, not just the embeddings, but with the POS window concatenated?
        x = T.concatenate((self.emb[self.idxs].reshape((self.idxs.shape[0], de*cs)),\
                           self.pos[self.pos_idxs].reshape((self.pos_idxs.shape[0],npos*cs))), 1)
        

        #self.y = T.matrix('y') # NB old code that should work. batch of sequence of vector (should be 0 when X is not null) 
        #self.y = T.ivector('y') #labels for whole sentence
        self.y = T.iscalar('y')
        #if single_output:
            #self.y = T.vector('y') #old code that should work.
        #    self.y = T.iscalar('y') # label
        self.h0 = shared(np.zeros(shape=self.n_lstm, dtype=dtype)) # initial hidden state         
        self.c0 = shared(np.zeros(shape=self.n_lstm, dtype=dtype)) # initial hidden state         
        #NB changing from using shared to not
        #self.lr = shared(np.cast[dtype](lr)) #learning rate
        #self.lr = np.cast[dtype](lr) #learning rate
        self.lr = T.scalar('lr')
        
        [h_vals, c_vals, y_vals], _ = theano.scan(fn=step_lstm,        
                                          sequences=x,
                                          outputs_info=[self.h0, self.c0, None],
                                          n_steps=x.shape[0])
        self.output = y_vals
        p_y_given_x_lastword = self.output[-1,0,:] #TODO does this work?
        p_y_given_x_sentence = self.output[:,0,:] #TODO does this work?
        p_y_given_x_sentence_hidden = (h_vals,c_vals,self.output[:,0,:])
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        
        #TODO adding this- zero one loss for the last word
        y_pred_word = T.argmax(p_y_given_x_lastword)
        
        #if single_output:
        #    self.output = y_vals[-1]            
        #else:
        #    self.output = y_vals
        
        self.cxe = T.mean(T.nnet.binary_crossentropy(self.output, self.y))
        #self.nll = -T.mean(self.y * T.log(self.output)+ (1.- self.y) * T.log(1. - self.output))     
        #TODO changing to below, same loss function
        self.nll = -T.mean(T.log(p_y_given_x_lastword)[self.y])
        self.mse = T.mean((self.output - self.y) ** 2)
        
        self.sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), self.y])
        
        
        self.L2_sqr =  sum([(p ** 2).sum() for p in self.params])
                        
        self.cost = self.nll + self.L2_reg * self.L2_sqr
        if cost_function == 'mse':
            self.cost = self.mse + self.L2_reg * self.L2_sqr
        elif cost_function == 'cxe':
            self.cost = self.cxe + self.L2_reg * self.L2_sqr
        
        #if na == 0: #assume no acoustic input for now
        self.classify = theano.function(inputs=[self.idxs,self.pos_idxs], outputs = y_pred)
        self.debug = theano.function(inputs = [x, self.y], outputs = [x.shape, self.y.shape, y_vals.shape, self.cost.shape])
        
        gradients = T.grad(self.cost, self.params)
        #self.updates = OrderedDict()
        #for param, gparam in zip(self.params, self.gparams):
        #    self.updates[param] = param - gparam * self.lr
        self.updates = OrderedDict(( p, p-self.lr*g ) for p, g in zip( self.params , gradients))
        
        
        self.loss = theano.function(inputs = [x, self.y], outputs = self.cost)
        #TODO getting rid of this:
        #self.train = theano.function(inputs = [x, self.y], outputs = self.cost, updates=self.updates)
        
        
        
        # theano functions #NB added POS
        #if na == 0: #assume no acoustic features for now
        self.soft_max = theano.function(inputs=[self.idxs, self.pos_idxs], outputs=p_y_given_x_sentence) #simply outputs the soft_max distribution for each word in utterance
        self.soft_max_return_hidden_layer = theano.function(inputs=[self.idxs, self.pos_idxs], outputs=p_y_given_x_sentence_hidden)
        
        #if na == 0: #assume no acoustic features for now
        self.train = theano.function( inputs  = [self.idxs, self.pos_idxs, self.y, self.lr],
                                   outputs = self.cost,
                                   updates = self.updates )
        
        #nb adding embeddings
        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})
        
        
        
    def fit(self, dialogues, lr, acoustic=True, load_data=True):
        """Fit method that takes pickled numpy matrices pathed to in the list
        files as its input.
        """
        print "training"
        print "acoustic", acoustic
        print "load data", load_data
        loss = 0.0 #will increment the loss as we go along
        tic = time.time()
        current_index = 0
        for i in range(0,len(dialogues)):
            data = dialogues[i][1]
            if load_data: #set to False on Bender for training as these will be python np objects there is enough space?
                data = np.load(data) #load the pickled numpy array
                _, acoustic_data, lex_data, pos_data, indices, labels = load_data_from_array(data, self.n_acoust)
            else:
                _, acoustic_data, lex_data, pos_data, indices, labels = data #should be bundles up already
            nw = acoustic_data.shape[0] # number of examples in this dialogue
            #if acoustic:
            #shuffle([train_lex, train_y], s['seed'])
            #tic = time.time()
            #mycorpus, myb_indices = corpusToIndexedMatrix(lex_data, , s['bs']) #window size across number of words deep, gets matrix too
            #mylabels = np.asarray(list(itertools.chain(*train_y)), dtype='int32')
            
            
            test = 0
            #load in the data to shared vars, can use 'set value too'
            #self.lexical_data = self.shared_dataset(lex_data) #loads dummy data set as a shared variable
            #self.pos_data = self.shared_dataset(pos_data)
            #self.acoustic_data = data = self.shared_dataset(acoustic_data,dtype=theano.config.floatX)
            #self.labels = self.shared_dataset(labels)
            for start,stop in indices:
                current_index+=1
                test+=1 #TODO for testing
                #if test > 50: break #TODO for testing
                if acoustic:
                    #print 'acoustic raw', acoustic_data[start:stop+1,:].shape
                    #ac = np.asarray(acoustic_data[start:stop+1,:],dtype='float32')
                    #print 'acoustic', ac.shape
                    #print 'lexical', lex_data[start:stop+1,:].shape
                    #raw_input()
                    x = self.train(lex_data[start:stop+1,:],pos_data[start:stop+1,:],np.asarray(acoustic_data[start:stop+1,:],dtype='float32'),labels[stop],lr)
                else:
                    #print lex_data[start:stop+1,:]
                    #print pos_data[start:stop+1,:]
                    x = self.train(lex_data[start:stop+1,:],pos_data[start:stop+1,:],labels[stop],lr)
                    #raw_input()
                    
                loss+=x
                self.normalize()
                
                #print '[learning] >> %2.2f%%'%((stop+1)*100./nw),'of file {} / {}'.format(i+1,len(dialogues)),\
            print 'file {} / {}'.format(i+1,len(dialogues)),'completed in %.2f (sec) <<\r'%(time.time()-tic)
            sys.stdout.flush()
            print "current train_loss", loss/float(current_index)
            #break #TODO switch back when actually training
        return loss/float(current_index)
    
    
    
    
    def fitold(self, my_seq, my_indices, my_labels, lr, nw, pos=None, sentence=False):
        """The main training function over the corpus indexing into my_indices"""
        tic = time.time()
        corpus = self.shared_dataset(my_seq) #loads data set as a shared variable
        labels = self.shared_dataset(my_labels)
        if not pos == None:
            pos = self.shared_dataset(pos)
        #TODO new effort to index the shared vars
        batchstart = T.iscalar('batchstart')
        batchstop = T.iscalar('batchstop')
        if sentence == True:
            cost = self.sentence_nll
            updates = self.sentence_updates
        else:    
            cost = self.nll
            updates = self.updates
        if pos == None:
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
            #TODO have changed  self.y : labels[batchstop]},   
        train_loss = 0.0
        laststop = 0
        for start,stop in my_indices:
            laststop = stop
            
            #if i> 50: break
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
                
    def corpus_nll(self, my_seq, my_indices, my_labels, pos=None, sentence=False):
        """Computes the average loss (per word) for whole corpus"""
        # batch sizes is a list the same length as losses
        # which has size of each batch in it (all of sizefrom 1 to bptt limit). Computes with myindices
        #batch_sizes = [(my_indices[i][1]-my_indices[i][0])+1 for i in range(0,len(my_indices))] 
        #print len(batch_sizes) #TODO don't actually need this for weighting the loss as this done anyway I believe by length of context sequence?

        corpus = self.shared_dataset(my_seq) #loads data set as a shared variable
        labels = self.shared_dataset(my_labels)
        if not pos == None:
            pos = self.shared_dataset(pos)
        
        batchstart = T.iscalar('batchstart')
        batchstop = T.iscalar('batchstop')
        if sentence == True:
            cost = self.sentence_nll
        else:    
            cost = self.nll
        
        if pos == None:
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
    
     
    def shared_dataset(self, mycorpus, borrow=True):
        """ Load the dataset into shared variables """
        return theano.shared(np.asarray(mycorpus, dtype='int32'),
                                 borrow=True)
    
    def load_weights_from_folder(self,folder):
        for name, param in zip(self.names, self.params):
            param.set_value(np.load(os.path.join(folder, name + ".npy")))
    
    def load_weights(self, emb=None, c0=None, h0=None):
        if not emb == None: self.emb.set_value(emb)
        if not c0 == None: self.c0.set_value(c0)
        if not h0 == None: self.h0.set_value(h0)
            
            
    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())


class LstmMiniBatch:
    def __init__(self, n_in, n_lstm, n_out, lr=0.05, batch_size=64, single_output=True, output_activation=T.nnet.softmax, cost_function='nll'):        
        self.n_in = n_in
        self.n_lstm = n_lstm
        self.n_out = n_out
        self.W_xi = init_weight((self.n_in, self.n_lstm),'W_xi') 
        self.W_hi = init_weight((self.n_lstm, self.n_lstm),'W_hi', 'svd') 
        self.W_ci = init_weight((self.n_lstm, self.n_lstm),'W_ci', 'svd') 
        self.b_i = shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_lstm)))
        self.W_xf = init_weight((self.n_in, self.n_lstm),'W_xf') 
        self.W_hf = init_weight((self.n_lstm, self.n_lstm),'W_hf', 'svd') 
        self.W_cf = init_weight((self.n_lstm, self.n_lstm),'W_cf', 'svd') 
        self.b_f = shared(np.cast[dtype](np.random.uniform(0, 1.,size = n_lstm)))
        self.W_xc = init_weight((self.n_in, self.n_lstm),'W_xc') 
        self.W_hc = init_weight((self.n_lstm, self.n_lstm),'W_hc', 'svd') 
        self.b_c = shared(np.zeros(n_lstm, dtype=dtype))
        self.W_xo = init_weight((self.n_in, self.n_lstm),'W_xo') 
        self.W_ho = init_weight((self.n_lstm, self.n_lstm),'W_ho', 'svd') 
        self.W_co = init_weight((self.n_lstm, self.n_lstm),'W_co', 'svd') 
        self.b_o = shared(np.cast[dtype](np.random.uniform(-0.5,.5,size = n_lstm)))
        self.W_hy = init_weight((self.n_lstm, self.n_out),'W_hy') 
        self.b_y = shared(np.zeros(n_out, dtype=dtype))
        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i, 
                       self.W_xf, self.W_hf, self.W_cf, self.b_f, 
                       self.W_xc, self.W_hc, self.b_c, 
                       self.W_ho, self.W_co, self.W_co, self.b_o, 
                       self.W_hy, self.b_y]
                

        def step_lstm(x_t, h_tm1, c_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + T.dot(c_tm1, self.W_ci) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + T.dot(c_tm1, self.W_cf) + self.b_f)
            c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc) + self.b_c) 
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo)+ T.dot(h_tm1, self.W_ho) + T.dot(c_t, self.W_co)  + self.b_o)
            h_t = o_t * T.tanh(c_t)
            y_t = T.nnet.softmax(T.dot(h_t, self.W_hy) + self.b_y) 
            return [h_t, c_t, y_t]

        X = T.tensor3() # batch of sequence of vector
        Y = T.tensor3() # batch of sequence of vector (should be 0 when X is not null) 
        h0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state         
        c0 = shared(np.zeros(shape=(batch_size,self.n_lstm), dtype=dtype)) # initial hidden state         
        self.lr = shared(np.cast[dtype](lr))
        
        [h_vals, c_vals, y_vals], _ = theano.scan(fn=step_lstm,        
                                          sequences=X.dimshuffle(1,0,2),
                                          outputs_info=[h0, c0, None])

        if single_output:
            self.output = y_vals[-1]            
        else:
            self.output = y_vals.dimshuffle(1,0,2)
        
        cxe = T.mean(T.nnet.binary_crossentropy(self.output, Y))
        nll = -T.mean(Y * T.log(self.output)+ (1.- Y) * T.log(1. - self.output))     
        mse = T.mean((self.output - Y) ** 2)

        cost = 0
        if cost_function == 'mse':
            cost = mse
        elif cost_function == 'cxe':
            cost = cxe
        else:
            cost = nll 
        
        gparams = T.grad(cost, self.params)
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * self.lr
        
        self.loss = theano.function(inputs = [X, Y], outputs = [cxe, mse, cost])
        self.train = theano.function(inputs = [X, Y], outputs = cost, updates=updates)
        self.predictions = theano.function(inputs = [X], outputs = y_vals.dimshuffle(1,0,2))
        self.debug = theano.function(inputs = [X, Y], outputs = [X.shape, Y.shape, y_vals.shape, cxe.shape])
