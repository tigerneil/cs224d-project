# cs224d-project


TODO
6/6/15
- Train baseline and recurrent on 100k set to compare with recursive
- Train a separate model to predict relation/no relation and integrate with baseline and recurrent for kbp pipeline
- Generate plots from the data in rnn_xxx_dev_perf.log

5/26/15:
- Melvin: concat 2 entity vectors, predict softmax of relation at the top
- Melvin: avg of trained word vectors, predict relation at the top
- Ankur: recurrent NN, using dep path
- Mikhail: recursive NN (hw3 implementation); use NLTK to get constituency parse for sentences (but only keep the subtree that contains the mentions)

5/17/15:
- Compute word2vec for KBP corpus
- Keras - implement basic MLP with triples
- Keras - implement average of word vectors
- Keras - implement average of words in dependency path
- Figure out about the training and test data mismatch



5/10/15:
- mikhail: batch_size = 20, lr = 0.001, reg = 0.001
- ankur: batch_size = 20, lr = 0.01, reg = 0.001
- melvin: batch_size = 20, lr = 0.01, reg = 0.01
- train average model on 1 million sentences and test using validation set
- utilities to save and load a model

5/3/15:
- DL word vectors (glove)
- get code to extract dep path between mentions
- generate micro datasets in addition to the 10k (maybe 100k, 1mil)
- write a recursive NN


Desc of Training data CSV:

 gloss                  | text     |           | extended | 
 dependencies_conll     | text     |           | extended | 
 words                  | text[]   |           | extended | 
 lemmas                 | text[]   |           | extended | 
 pos_tags               | text[]   |           | extended | 
 ner_tags               | text[]   |           | extended | 
 subject_id             | bigint   |           | plain    | 
 subject_entity         | text     |           | extended | 
 subject_link_score     | real     |           | plain    | 
 subject_ner            | text     |           | extended | 
 object_id              | bigint   |           | plain    | 
 object_entity          | text     |           | extended | 
 object_link_score      | real     |           | plain    | 
 object_ner             | text     |           | extended | 
 subject_begin          | smallint |           | plain    | 
 subject_end            | smallint |           | plain    | 
 object_begin           | smallint |           | plain    | 
 object_end             | smallint |           | plain    | 
 known_relations        | text[]   |           | extended | 
 incompatible_relations | text[]   |           | extended | 
 annotated_relation     | text     |           | extended | 


Desc of Test data CSV:

       Column       |   Type   | Modifiers | Storage  | Description 
--------------------+----------+-----------+----------+-------------
 gloss              | text     |           | extended | 
 dependencies_conll | text     |           | extended | 
 words              | text[]   |           | extended | 
 lemmas             | text[]   |           | extended | 
 pos_tags           | text[]   |           | extended | 
 ner_tags           | text[]   |           | extended | 
 subject_id         | bigint   |           | plain    | 
 subject_entity     | text     |           | extended | 
 subject_link_score | real     |           | plain    | 
 subject_ner        | text     |           | extended | 
 object_id          | bigint   |           | plain    | 
 object_entity      | text     |           | extended | 
 object_link_score  | real     |           | plain    | 
 object_ner         | text     |           | extended | 
 subject_begin      | smallint |           | plain    | 
 subject_end        | smallint |           | plain    | 
 object_begin       | smallint |           | plain    | 
 object_end         | smallint |           | plain    | 


