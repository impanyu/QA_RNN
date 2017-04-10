# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib import rnn 

import time
import operator

import numpy as np
import tensorflow as tf
import fileinput 

import re
import numpy as np
import sys

import json
import QAreader


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "./model_weights",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool(
    "ifcontinue",True,
    "If continuing to learn or start from scratch.")
flags.DEFINE_string(
    "data_type","1",
    "Which data set you want to train the model on.")
FLAGS = flags.FLAGS




def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data,vocab=1000, name=None,answering=False):
    #self.data=data
    self.batch_size = batch_size = config.batch_size
    #self.num_steps = num_steps = config.num_steps
    if (answering):
        self.epoch_size=1
    else:
        self.epoch_size = (len(data[0]) // batch_size) 
    #print(data[0])
    self.documents, self.questions,self.vanswers,self.answers,_= QAreader.ptb_producer(data[0],data[1],data[2], batch_size,vocab,name=name,config=config)
    #self.epoch_size= self.documents.get_shape().as_list()[0] // batch_size
    #print( self.documents.get_shape().as_list())
    #print (self.questions)
    
class PTBModel(object):
  """The PTB model."""


  def __init__(self, is_training, config, input_,answering=False):

    '''
    def BiRNN(x):

      # Prepare data shape to match `bidirectional_rnn` function requirements
      # Current data input shape: (batch_size, n_steps, n_input)
      # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

      # Permuting batch_size and n_steps
      x = tf.transpose(x, [1, 0, 2])
      # Reshape to (n_steps*batch_size, n_input)
      x = tf.reshape(x, [-1, size])
      # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
      x = tf.split(x, num_steps, 0)

      # Define lstm cells with tensorflow
      # Forward direction cell
      lstm_fw_cell = rnn.BasicLSTMCell(size, forget_bias=1.0)
      # Backward direction cell
      lstm_bw_cell = rnn.BasicLSTMCell(size, forget_bias=1.0)

      # Get lstm cell output
      try:
          outputs, output_state_fw, output_state_bw = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                dtype=tf.float32)
      except Exception: # Old TensorFlow version only returns outputs not states
          outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                          dtype=tf.float32)

      # Linear activation, using rnn inner loop last output
      return  outputs,output_state_fw,output_state_bw #tf.matmul(outputs[-1], weights['out']) + biases['out']
    '''

    self._input = input_
    documents_input =  input_.documents
    questions_input = input_.questions
    self.answers_input = input_.answers
    #print(questions_input.get_shape())
    batch_size = input_.batch_size
    document_steps = documents_input.get_shape().as_list()[1]
    question_steps = questions_input.get_shape().as_list()[1]
    self.document_steps=document_steps
    self.question_steps=question_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())
    #(self._initial_state)

    
    
    
    #with tf.device("/cpu:0"):
    #embedding = tf.get_variable(
     #    "embedding", [vocab_size, size], dtype=data_type())
    embedding= tf.Variable(
        tf.random_uniform([vocab_size, size], -1.0, 1.0),name="embedding",dtype=data_type())
    self.embedding =embedding.name
    print(embedding.name)
    documents = tf.nn.embedding_lookup(embedding, input_.documents)
    questions = tf.nn.embedding_lookup(embedding, input_.questions)
    
      #answers = tf.nn.embedding_lookup(embedding, input_.answers)
        
    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.nn.rnn(cell, inputs,
    #                            initial_state=self._initial_state)
    
    print("1")
  
    doc_outputs=[]
    cell_output_fws=[]
    cell_output_bws=[]
    documents_reverse= tf.reverse(documents,[1])
    questions_reverse= tf.reverse(questions,[1])
    doc_weights=[]
    doc_avg=[]
    #que_outputs=[]
    print(document_steps)
    with tf.variable_scope("documents"):
        #doc_outputs: time_step,batch_size,2*size
        #compute document output
        state_fw = self._initial_state
        state_bw = self._initial_state
        for time_step in range(document_steps):
            #print(time_step)
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output_fw, state_fw)=cell(documents[:, time_step, :], state_fw,scope="doc_fw")
            (cell_output_bw, state_bw) = cell(documents_reverse[:, time_step, :],state_bw,scope="doc_bw")
            cell_output_fws.append(cell_output_fw)
            cell_output_bws.append(cell_output_bw)
        for time_step in range(document_steps):   
            doc_outputs.append(tf.concat([cell_output_fws[time_step],tf.reverse(cell_output_bws,[0])[time_step]],1))
        doc_outputs=tf.convert_to_tensor(doc_outputs)
      
       # tf.get_variable_scope().reuse=None
        #que_output: batch_size,2*size
    print("2")
    with tf.variable_scope("questions"):  
        #compute question output
        state_fw = self._initial_state
        
        state_bw = self._initial_state
        for time_step in range(question_steps):
            
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output_fw, state_fw)= cell(questions[:, time_step, :], state_fw,scope="que_fw")
            (cell_output_bw, state_bw) = cell(questions_reverse[:, time_step, :],state_bw,scope="que_bw")
        #que_output=tf.concat([cell_output_fws[question_steps-1],tf.reverse(cell_output_bws,[1])[question_steps-1]],1)
        #print(document_steps)
        que_output=tf.concat([cell_output_fw,cell_output_bw],1)
        #doc_weights : time_step,batch_size
        '''
        for time_step in range(document_steps): 
            doc_weights.append([])
            temp=tf.matmul(doc_outputs[time_step],tf.transpose(que_output))
        
        '''
    matrix_w=tf.get_variable(
        "W", [2*size, 2*size], dtype=data_type())
    
    for batch in range(batch_size): 
          
          temp_vector=tf.matmul(tf.matmul(doc_outputs[:,batch,:],matrix_w  ),  tf.reshape(que_output[batch,:],[2*size,1])  )
          doc_weights.append(tf.nn.softmax( temp_vector ,0) )
    doc_weights=tf.convert_to_tensor(doc_weights)
    doc_weights= tf.transpose(tf.reshape( doc_weights,[batch_size,document_steps]))
    #doc_weights=tf.nn.softmax(doc_weights,0)
    self.doc_weights=doc_weights
    #doc_choice=tf.argmax(doc_weights,1)
    #doc_choice=tf.cast(doc_choice,tf.int32)
    logits=[]
    for batch in range(batch_size): 
        tmp=tf.one_hot(input_.documents[batch], config.vocab_size, on_value=1.0, off_value=0.0, axis=-1, dtype=data_type())
        tmp=tf.transpose(tmp)
        #tmp:vocab_size,time_step
        tmp2=tf.matmul(tmp,tf.reshape(doc_weights[:,batch],[document_steps,1]))
        #tmp2:vocab_size,1
        tmp2=tf.reshape(tmp2,[vocab_size])
        logits.append(tmp2)
        #logits.append(input_.documents[batch][doc_choice[batch]])
    #logits: batch_size,vocab_size
    logits=tf.convert_to_tensor(logits)
    #logits=tf.one_hot(logits, config.vocab_size, on_value=1.0, off_value=0.0, axis=-1, dtype=data_type())
        
        #doc_avg:batch_size,2*size  
        # temp=tf.matmul(tf.transpose(doc_weights),doc_outputs)
        
    '''
    for batch in range(batch_size): 
            
            doc_avg.append(tf.reshape( tf.matmul(tf.reshape(doc_weights[:,batch],[1,document_steps]),doc_outputs[:,batch,:]),[2*size]))
    doc_avg=tf.convert_to_tensor(doc_avg)    
    
    self.doc_avg=doc_avg[0]      
    '''    
        #final_g: batch_size,4*size
    #final_g=tf.concat([doc_avg,que_output],1)
                
    print("3")            
                
    #outputs = (np.array(outputs)[:,0:len(outputs[0])//2-1]).tolist()
  

    #for i in range(len(outputs)):
     # outputs[i] = outputs[i][:,0:size]
   
    # inputs: batch_size,num_steps,size
    # outputs is a list of two dimensional tensors
    # others are all tensors
    # outputs: num_steps,batch_size,size
    # output: num_steps*batch_size, size
    # logits: num_steps*batch_size,vocab_size
    # input_.targets: batch_size, num_steps
  

    #  output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
    '''
    softmax_w = tf.get_variable(
        "softmax_w", [2*size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(doc_avg, softmax_w) + softmax_b
    self.softmax_b=softmax_b
    '''
    '''
    softmax_w2 = tf.get_variable(
        "softmax_w2", [vocab_size, vocab_size], dtype=data_type())
    softmax_b2 = tf.get_variable("softmax_b2", [vocab_size], dtype=data_type())
    logits = tf.matmul(tf.nn.softmax(mid), softmax_w2) + softmax_b2
    '''
    '''
    self.softmax_w=softmax_w
    '''
    #print(tf.nn.softmax(logits)[0].get_shape())
    self.word_index=tf.argmax( tf.nn.softmax(logits)[0],0)
    
    #self.answers_input=tf.reshape(self.answers_input,[20,1])
    self.logits_origin=logits[0]
    self.logits = tf.nn.softmax(logits[0])#tf.divide(tf.exp(logits[0]), tf.reduce_sum(tf.exp(logits[0])))
    self.vanswer=input_.vanswers[0]
    #tf.cast(input_.vanswers,data_type())
    loss=tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=input_.vanswers, logits=logits))
    self._cost=cost=loss
    self.first_loss=tf.nn.softmax_cross_entropy_with_logits(labels=tf.cast(input_.vanswers,data_type()), logits=logits)[0]
    #tf.reduce_sum(tf.cast(input_.vanswers[0], tf.float32) * tf.log(tf.nn.softmax(logits[0]) ))  #
    
    '''
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.answers, [-1])],
        [tf.ones([batch_size], dtype=data_type())])
  
    #print(type(loss))
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    '''
    #self._final_state = state
    self.actual=input_.answers[0]
    
    
    self.correct_prediction = tf.reduce_mean( tf.cast(tf.equal(tf.argmax(logits,1),tf.cast(input_.answers,tf.int64)), data_type()) )
    #print(self.actual.get_shape())
    print("2")
    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)

    self.train_step = tf.train.GradientDescentOptimizer(self._lr).minimize(loss)
    '''
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
    '''
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.1
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size =300
  max_epoch = 4
  max_max_epoch = 10
  keep_prob = 1
  lr_decay = 0.6
  batch_size = 20
  vocab_size = 100


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False,training=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "logits":model.logits,
     # "final_state": model.final_state,
      "index":model.word_index,
      "actual":model.actual,
      "vanswer":model.vanswer,
      "first_loss":model.first_loss,
      #"doc_avg":model.doc_avg,
      
      "logits_origin":model.logits_origin,
      "correct_prediction":model.correct_prediction,
      "embedding": model.embedding,
      
      "initial_state":model._initial_state,
      "doc_weights":model.doc_weights
  }
  if training:
    fetches["train"]=model.train_step
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    #print("inside")
    state = session.run(model.initial_state)
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    #state = vals["final_state"]
    
    #print(vals["initial_state"])
    index=vals["index"]
    #print(index)
    #print(vals["softmax_w"])
    
    
    #print(vals["doc_avg"])
    
    #print(vals["logits"])
    #print(vals["logits_origin"])
    #print(vals["vanswer"])
    #print(vals["first_loss"])
    #print(vals["doc_weights"])
    print("Accuracy in this round: %s"  % vals["correct_prediction"])
    answer_word=QAreader._word_id_to_word(index)
    print(answer_word)
    print("actual word: %s" % QAreader._word_id_to_word (vals["actual"]))
    #print(vals["actual"])
    '''
    print(fetches["logits"].get_shape())

    print(type(logits)) 
    if (verbose):
         index, _ =logits[0].argmax()
         answer_word=QAreader._word_id_to_word(index)
         print(answer_word)
    '''         
            
    costs += cost
    iters += model.document_steps+model.question_steps

    if verbose and step % 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size,cost, #cost,#np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))
     # print("%s" % fetches["answer"])

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

    


def main(_):
    
    
  answering_data=[]
  terminate=False
  answering_documents=[]
  answering_answers=[]
  answering_questions=[]

  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  train_data, test_data,vocab = QAreader.prepare_data(FLAGS.data_path,FLAGS.data_type)

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1
  #config.vocab_size=vocab
  #eval_config.vocab_size=vocab

  with tf.Graph().as_default():
        
   
    if FLAGS.ifcontinue==False:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
    else:
        initializer = None
        #config.learning_rate=0.01

    with tf.name_scope("Train"):
        train_input = PTBInput(config=config,vocab=vocab, data=train_data, name="TrainInput")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input)
        tf.summary.scalar("Training Loss", m.cost)
        tf.summary.scalar("Learning Rate", m.lr)

        '''
        with tf.name_scope("Valid"):
          valid_input = PTBInput(config=config, data=test_data,vocab=vocab, name="ValidInput")
          with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
          tf.summary.scalar("Validation Loss", mvalid.cost)


        with tf.name_scope("Test"):
          test_input = PTBInput(config=eval_config, vocab=vocab,data=test_data, name="TestInput")
          with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config,
                             input_=test_input)
        '''    
    print("model built")

    #saver = tf.train.Saver()          
    sv = tf.train.Supervisor()#(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
    #with tf.Session() as sess:
      #tf.global_variables_initializer().run()
        
      if FLAGS.ifcontinue:
         #new_saver = tf.train.import_meta_graph('my-model.meta')
         sv.saver.restore(session,  "./model_weights_"+FLAGS.data_type)
         print("model restored!")
      for mm in tf.global_variables():
         print (mm.name)
    #with tf.Session() as session:
      for i in range(config.max_max_epoch):
        if FLAGS.ifcontinue==False:
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            #lr_decay = config.lr_decay 
            #lr_decay=lr_decay*0.5
            m.assign_lr(session, config.learning_rate * lr_decay)
        else:
            #lr_decay = config.lr_decay ** max(config.max_max_epoch - config.max_epoch, 0.0)
            #lr_decay=1
            #config.learning_rate=0.0005
            m.assign_lr(session, 0.0005)
       
        print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m , verbose=True,training=True)   #, eval_op=m.train_op,
                                    
        print("Epoch: %d Train Perplexity: %.5f" % (i + 1, train_perplexity))
        #valid_perplexity = run_epoch(session, mvalid)
        #print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        
      #test_perplexity = run_epoch(session, mtest)
      #print("Test Perplexity: %.3f" % test_perplexity)
       
      if FLAGS.save_path:
        print("Saving model to %s." %  "./model_weights_"+FLAGS.data_type)
        print(sv.saver.save(session,  "./model_weights_"+FLAGS.data_type))
        #saver.save(session, "./model.ckpt")
   

 
   
        
      
if __name__ == "__main__":
  tf.app.run()
