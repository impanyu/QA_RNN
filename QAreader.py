#copyright 2015 The TensorFlow Authors. All Rights Reserved.
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


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re
import tensorflow as tf

import json


word_to_id={}
def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _read_file(dir,ifTrain,data_type):
    documents=[]
    questions=[]
    answers=[]
    tuples=[]
    currentDocument=[]
    maxq=0;
    maxd=0;
    if(ifTrain):
      train="train"
    else:
      train="test"
    for filename in os.listdir(dir):
      #print(filename)
      if(filename[3]=="_"):
        c=(int)(filename[2])#question class
      else:   
        c=(int)(filename[2:4])
      

      if (filename.endswith(train+".txt") and filename.startswith("qa"+data_type+"_")): #and filename[3]=="_"): 
        with tf.gfile.GFile(os.path.join(dir, filename), "r") as f:
           
            for newline in f:
                
                newline=re.sub("\?|\.|,","",newline.lower())
                newarray=newline.split()
                if(newarray[0]=="1"):# a new document
                    newdoc=True
                    
                    newline=re.sub("[0-9]|[0-9][0-9]"," ",newline)
                    currentDocument=newline.split()
                elif(re.match( "[0-9]|[0-9][0-9]", newarray[len(newarray)-1])==None):#continuing a document 
                    newline=re.sub("[0-9]|[0-9][0-9]"," ",newline)
                    currentDocument+=newline.split()
                else:#find a document/question/answer tuple
                    newline=re.sub("[0-9]|[0-9][0-9]"," ",newline)
                    if(len(currentDocument)>200):# ignore those cases with too long documents
                        continue
                    
                    documents.append(currentDocument[:])
                    #print( documents)
                    newarray=newline.split()
                    answers.append(newarray[len(newarray)-1])
                    questions.append(newarray[0:len(newarray)-1])
                    tuples.append(newdoc)
                    newdoc=False
                    if(len(newarray)-1>maxq):
                        maxq=len(newarray)-1
                    if(len(currentDocument)>maxd):
                        maxd=len(currentDocument)

    #print(documents[0:10])
    #do some padding
    
    for i in range(len(documents)):
        if(tuples[i]):
            currentPad=documents[i] # the first documents in a series used as padding     
        documents[i]=currentPad*((maxd-len(documents[i]))//len(currentPad))+ documents[i]#pad in the beginning of each document
     
    for d in documents:
        d+=(maxd-len(d))* ["PAD"]
        #print(d)
    
 
   
    for q in questions:
        q+=(maxq-len(q))* ["PAD"] 
    
    return documents,questions,answers,tuples,maxq,maxd
    
                
                
                
def _build_vocab(filename):
  global word_to_id
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  #return word_to_id


def _build_vocab2(data):

  global word_to_id
  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  print(word_to_id)
  
 
  # save to file:
  with open('./word_to_id.json', 'w') as f:
      json.dump(word_to_id, f)
  ''' 
  #return word_to_id
  '''
  return len(words)

def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

def _words_to_word_ids(words, word_to_id):
  #(words)
  return [word_to_id[word] for word in words if word in word_to_id]

def _word_id_to_word(id):
    temp=sorted(word_to_id.items(),key=lambda x:(x[1],x[0]))
    words, _ = list(zip(*temp))
    if(id<len(words)):
        return words[id]
    else:
        print(id)
        return "unkown"
    
    
all_words=[]

def prepare_data(data_path=None,data_type="1"):
  global all_words
  global word_to_id
  trainDocuments,trainQuestions,trainAnswers,trainTuples,_,_=_read_file(data_path,True,data_type)
  testDocuments,testQuestions,testAnswers,testTuples,_,_=_read_file(data_path,False,data_type)



  print("raw data reading complete")
  print(len(trainDocuments))
  print(trainDocuments[0:10])
  
  for document in trainDocuments:
    all_words=all_words+document
  for question in trainQuestions:
    all_words+=question
  for answer in trainAnswers:
    #print (answer)
    all_words.append(answer)
  
 
  for document in testDocuments:
    all_words+=document
  for question in testQuestions:
    all_words+=question
  for answer in testAnswers:
    all_words.append(answer)
    
  vocab=_build_vocab2(all_words)
  print("vocab size is %i" % vocab)


  for i in range(len(trainDocuments)):
    trainDocuments[i]=_words_to_word_ids(trainDocuments[i],word_to_id)
  for i in range(len(trainQuestions)):
    trainQuestions[i]=_words_to_word_ids(trainQuestions[i],word_to_id)
  
  trainAnswers=_words_to_word_ids(trainAnswers,word_to_id)

  for i in range(len(testDocuments)):
    testDocuments[i]=_words_to_word_ids(testDocuments[i],word_to_id)
  for i in range(len(testQuestions)):
    testQuestions[i]=_words_to_word_ids(testQuestions[i],word_to_id)
  
  testAnswers=_words_to_word_ids(testAnswers,word_to_id)
  
  train_data=[]
  test_data=[]
  train_data.append(trainDocuments)
  train_data.append(trainQuestions)
  train_data.append(trainAnswers)
  train_data.append(trainTuples)

  test_data.append(testDocuments)
  test_data.append(testQuestions)
  test_data.append(testAnswers)
  test_data.append(testTuples)

  print("data prepare complete")

  return train_data,test_data,vocab

def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTbB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)


  train_data = _words_to_word_ids(train_path, word_to_id)
  valid_data = _words_to_word_ids(valid_path, word_to_id)
  test_data = _words_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary



def ptb_producer(doc,que,ans, batch_size, vocab=100,name=None,config=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  #print(ans)
  with tf.name_scope(name, "PTBProducer", [doc,que, ans]):
    doc_len=len(doc)
    vocab=config.vocab_size
    vans=[]
    for e in ans:
        van=[0]*vocab
        van[e]=1
        vans.append(van)
    #print(doc)
    d=len(doc[0])
    q=len(que[0])
    
    #print(d)
    
    epoch_size = doc_len // batch_size
    #print(epoch_size)
    #print(ans)
    
    doc = tf.convert_to_tensor(doc, name="documents", dtype=tf.int32)
    que = tf.convert_to_tensor(que, name="questions", dtype=tf.int32)
    vans = tf.convert_to_tensor(vans, name="vanswers", dtype=tf.int32)
    ans = tf.convert_to_tensor(ans, name="answers", dtype=tf.int32)
    
    #data_len = len(documents)
    # batch_len = len(documents[0])

   
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    
    
    #elems = tf.convert_to_tensor([1,2,3,5])
    batch_prob=[]
    #for batch_number in range(batch_size):
    batch_prob.append([10.]*doc_len)
        
    samples = tf.multinomial(tf.log(batch_prob), batch_size) # note log-prob
    print(tf.get_variable_scope().reuse == False)
    x=[]
    y=[]
    z=[]
    zz=[]
    for batch_number in range(batch_size):
        x.append(doc[tf.cast(samples[0][batch_number], tf.int32)])
        y.append(que[tf.cast(samples[0][batch_number], tf.int32)])
        z.append(vans[tf.cast(samples[0][batch_number], tf.int32)])
        zz.append(ans[tf.cast(samples[0][batch_number], tf.int32)])
        
    x= tf.convert_to_tensor(x, name="documents", dtype=tf.int32)
    y = tf.convert_to_tensor(y, name="questions", dtype=tf.int32)
    z = tf.convert_to_tensor(z, name="vanswers", dtype=tf.int32)
    zz = tf.convert_to_tensor(zz, name="answers", dtype=tf.int32)
    '''
    x= tf.slice(doc,[i*batch_size,0],[batch_size,d])
    y= tf.slice(que,[i*batch_size,0],[batch_size,q])
    z= tf.slice(vans,[i*batch_size,0],[batch_size,vocab])
    zz=tf.slice(ans,[i*batch_size],[batch_size])
    '''
    #print(i)
    #print(epoch_size)
    return x, y, z ,zz,epoch_size


