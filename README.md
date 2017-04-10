# Teaching Machine to Answer Question Using Recurrent Neural Networks and Attentive Reader
=======================================================================
Tensorflow implementation of [Text Understanding with the Attention Sum Reader Network] https://arxiv.org/abs/1603.01547 with reference to
[Teaching Machines to Read and Comprehend] https://arxiv.org/abs/1506.03340
and [A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task] https://arxiv.org/abs/1606.02858

The dataset is based on the paper [Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks]
 https://arxiv.org/abs/1502.05698 
 
 Prerequisites
-------------

- Python 2.7+
- [Tensorflow 1.0](https://www.tensorflow.org/)
 
 
Usage
-----

To train a model from scratch (ifcontinue==False indicates this is a new model, data_type is the type of the tasks you want to train on and is between 1-20)

    $ python QAtrain_weights.py --data_path=./tasks_1-20_v1-2/en/ --model=small --ifcontinue=False --data_type=1
    
To train a model from scratch (ifcontinue==True indicates continuing to train an existing model, data_type is the type of the tasks you want to train on and is between 1-20)

   $ python QAtrain_weights.py --data_path=./tasks_1-20_v1-2/en/ --model=small --ifcontinue=False --data_type=1
   
To test an existing model (data_type is the type of the tasks and is between 1-20):

   $ python QAanswer.py --data_path=./tasks_1-20_v1-2/en/ --model=small --data_type=1


Credit
------

Modified codes based on TensorFlow official tutorial on RNN https://www.tensorflow.org/tutorials/recurrent

