
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
import sys

import QAtrain_weights


import QAreader
import json


FLAGS = QAtrain_weights.flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


session=None

def main(_):
    
  global session 

  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  #train_data, test_data=QAreader.prepare_data(FLAGS.data_path)

  config = QAtrain_weights.get_config()
  eval_config = QAtrain_weights.get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1



  while(True):
            answering_data=[]
            terminate=False
            answering_documents=[]
            answering_answers=[]
            answering_questions=[]
 
            if(terminate): break
            print("please input text and question:")
            currentDocument=[]
            newarray=[]
            pline=""
            for newline in  iter(sys.stdin.readline, ''):
                #print(newline)
                newline=re.sub("\?|\.|,|\n","",newline.lower())
                newline=re.sub("[0-9]|[0-9][0-9]"," ",newline)
                #print(type(newline))
                
                # newline=re.sub("[0-9]|[0-9][0-9]"," ",newline)

                #print(newarray)
                if(newline):                  
                   
                    
                    currentDocument+=pline.split()
                    pline=newline
                else:
                    
                    break
             
            if(len(currentDocument)==0): continue 
            newarray=pline.split()
            #if(len(newarray)==0): continue
            answering_documents.append(currentDocument+["PAD","PAD","PAD"])
            #answering_documents.append(currentDocument)
            
            answering_answers.append("PAD")
            #answering_answers.append("PAD")
            
            answering_questions.append(newarray[0:len(newarray)]+["PAD"])
            #answering_questions.append(newarray[0:len(newarray)])
            #print(answering_questions)

           
            
            all_words=[]
            for document in answering_documents:
                 all_words+=document
            for question in answering_questions:
                 all_words+=question
            for answer in answering_answers:
                 all_words.append( answer)
            #print(all_words)
            
            #eval_config.vocab_size=QAreader._build_vocab2(all_words)

            #eval_config.vocab_size=20
            
            '''
            QAreader._build_vocab2(all_words)
            '''
            with open('./word_to_id.json', 'r') as f:
                try:
                    QAreader.word_to_id = json.load(f)
                # if the file is empty the ValueError will be thrown
                except ValueError:
                    QAreader.word_to_id = {}   
                
            
            for i in range(len(answering_documents)):
                answering_documents[i]=QAreader._words_to_word_ids(answering_documents[i],QAreader.word_to_id)
            for i in range(len(answering_questions)):
                answering_questions[i]=QAreader._words_to_word_ids(answering_questions[i],QAreader.word_to_id)

            answering_answers=QAreader._words_to_word_ids(answering_answers,QAreader.word_to_id)
           # print(QAreader.word_to_id)
            


            answering_data.append(answering_documents)
            answering_data.append(answering_questions)
            answering_data.append(answering_answers)
            #print(answering_data[0])
  
 
            with tf.Graph().as_default():
                initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)

                with tf.name_scope("Train"):
                      answering_input = QAtrain_weights.PTBInput(config=eval_config,data=answering_data,                     name="AnsweringInput",answering=True)
                      with tf.variable_scope("Model", reuse=False, initializer=None):
                         manswering = QAtrain_weights.PTBModel(is_training=False, config=eval_config,
                                     input_=answering_input,answering=True)


                sv = tf.train.Supervisor()
             

                with sv.managed_session() as session:
                     # saver.restore(sess, FLAGS.save_path)
                      sv.saver.restore(session, "./model_weights_"+FLAGS.data_type)
                      print("try to answer")
                      answering_perplexity = QAtrain_weights.run_epoch(session, manswering,verbose=True)
         
            
        
      
if __name__ == "__main__":
  tf.app.run()
