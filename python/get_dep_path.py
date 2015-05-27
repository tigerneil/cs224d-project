#! /usr/bin/env python

"""
Extractor for relation mention features.

Outputs 2 feature for each relation mention:
  - the dependency path between the mentions
  - the presence of the words "wife", "widow", or "husband" along the dependency path
    (this should help with the spouse relation)

(refer to http://www.stanford.edu/~jurafsky/mintz.pdf)

Input query:

        SELECT s.doc_id AS doc_id,
               s.sentence_id AS sentence_id,
               array_to_string(max(s.lemma), '~^~') AS lemma,
               array_to_string(max(s.dep_graph), '~^~') AS dep_graph,
               array_to_string(max(s.words), '~^~') AS words,
               array_to_string(array_accum(m.mention_id), '~^~') AS mention_ids,
               array_to_string(array_accum(m.word), '~^~') AS mention_words,
               array_to_string(array_accum(m.type), '~^~') AS types,
               array_to_string(array_accum(m.start_pos), '~^~') AS starts,
               array_to_string(array_accum(m.end_pos), '~^~') AS ends
        FROM sentence s,
             mentions m
        WHERE s.doc_id = m.doc_id AND
              s.sentence_id = m.sentence_id
        GROUP BY s.doc_id,
                 s.sentence_id
"""

import sys
import dd as ddlib
import string
import re
import csv
import numpy as np
# the delimiter used to separate columns in the input
ARR_DELIM = ','


def dep_format_parser(dep_edge_str):
  """
  Given a string representing a dependency edge, return a tuple of
     (parent_index, edge_label, child_index).

  Args: dep_edge_str - a string representation of an edge in the dependency tree
             (e.g. "31 prep_of 33")
  Returns: tuple of (integer, string, integer) (e.g. (30, "prep_of", 32))
  """
  child, parent, label = dep_edge_str.split()
  return (int(parent)-1, label, int(child)-1) # input edge used 1-based indexing       

def get_recurrent_features(row):
  line = row.strip().split('\t')
  dep_graph_str = string.replace(line[1], '\\t', '\t')
  dep_graph_str = string.replace(dep_graph_str, '\\n', '\n')
  #dep_graph_str = string.replace(dep_graph_str, '\\\'', '\'')
  lemma_str = line[3]
  words_str = line[2]
  words_str = string.replace(words_str, "\",\"", "~^~")
  # skip sentences with empty dependency graphs
  #if dep_graph_str == "":
  #  return ""
  types = [line[9], line[13]]
  starts = [line[14], line[16]]
  ends = [line[15], line[17]]
  lemma = lemma_str.split(ARR_DELIM)
  dep_graph = dep_graph_str.split("\n")
  #PATTERN = re.compile(r'''((?:"[^"]*")+)''')
  #words = PATTERN.split(words_str[1:-1])[1::2]
  words = words_str.split(",")
  for i,word in enumerate(words):
	if word == "~^~":
		words[i] = ','
  mention_ids = [line[7], line[11]]
  mention_words = [[words[int(starts[0]): int(ends[0])]],[words[int(starts[1]):int(ends[1])]]]
  # create a list of mentions
  mentions = zip(mention_ids, mention_words, types, starts, ends)
  mentions = map(lambda x: {"mention_id" : x[0], "word" : x[1], "type" : x[2], "start" : int(x[3]), "end" : int(x[4])}, mentions)

  relation = None
  if len(line) == 21:
	relation = line[18]
  # get a list of Word objects
  obj = {}
  obj['lemma'] = lemma
  obj['words'] = words
  obj['dep_graph'] = dep_graph
  word_obj_list = ddlib.unpack_words(obj, lemma='lemma', words='words', dep_graph='dep_graph', dep_graph_parser=dep_format_parser)
  # at this point we have a list of the mentions in this sentence

  # go through all pairs of mentions
  for m1 in mentions:
    start1 = m1["start"]
    end1 = m1["end"]

    #if m1["type"] not in ["PERSON", "ORGANIZATION"]:
    #  continue

    for m2 in mentions:
      #if m1["mention_id"] == m2["mention_id"]:
        #continue

      start2 = m2["start"]
      end2 = m2["end"]

      edges = ddlib.dep_path_between_words(word_obj_list, end1 - 1, end2 - 1)
      #print edges
      if len(edges) > 0:
        num_roots = 0 # the number of root nodes
        num_left = 0 # the number of edges to the left of the root
        num_right = 0 # the number of edges to the right of the root
        left_path = "" # the dependency path to the left of the root
        right_path = "" # the dependency path to the right of the root

        # find the index of the switch from up to down
        switch_direction_index = -1
        for i in range(len(edges)):
          if not edges[i].is_bottom_up:
            switch_direction_index = i
            break
        
        # iterate through the edge list
        for i in range(len(edges)):
          curr_edge = edges[i]

          # count the number of roots; if there are more than 1 root then our dependency
          # path is disconnected
          if curr_edge.label == 'ROOT':
            num_roots += 1

          # going from the left to the root
          if curr_edge.is_bottom_up:
            num_left += 1

            # if this is the edge pointing to the root (word2 is the root)
            if i == switch_direction_index - 1:
              left_path = left_path + ("--" + curr_edge.label + "->")
              root = curr_edge.word2.lemma.lower()
	      #root = curr_edge.word2.word
            # this edge does not point to the root
            else:
              # if we are at the last edge, don't include the word (part of the mention)
              if i == len(edges) - 1:
                left_path = left_path + ("--" + curr_edge.label + "->")
              else:
                left_path = left_path + ("--" + curr_edge.label + "->" + curr_edge.word2.lemma.lower())
	        #left_path = left_path + ("--" + curr_edge.label + "->" + curr_edge.word2.word)

          # going from the root to the right
          else:
            num_right += 1

            # the first edge to the right of the root
            if i == switch_direction_index:
              right_path = right_path + "<-" + curr_edge.label + "--"
	      #right_path = right_path + "<-" + curr_edge.label + "--"

            # this edge does not point from the root
            else:
              # if we are at the first edge, don't include the word (part of the mention)
              if i == 0:
                right_path = right_path + ("<-" + curr_edge.label + "--")
              else:
                # word1 is the parent for right to left
                right_path = right_path + (curr_edge.word1.lemma.lower() + "<-" + curr_edge.label + "--")
		#right_path = right_path + (curr_edge.word1.word + "<-" + curr_edge.label + "--")

        # if the root is at the end or at the beginning (direction was all up or all down)
        if num_right == 0:
          root = "|SAMEPATH"
        elif num_left == 0:
          root = "SAMEPATH|"

        # if the edges have a disconnect
        elif num_roots > 1:
          root = "|NONEROOT|"

        # this is a normal tree with a connected root in the middle
        else:
          root = "|" + root + "|"

        path = left_path + root + right_path

        feat = [m1["word"], m2["word"], m1["type"], m2["type"], path]
        # make sure each of the strings we will output is encoded as utf-8
	if relation is not None:
		feat.append(relation[1:-1])
	return feat
  return [m1["word"], m2["word"], m1["type"], m2["type"], ""]

def get_recurrent_features_new(row):
  #print row
  line = row.strip().split('\t')
  dep_graph_str = string.replace(line[1], '\\t', '\t')
  dep_graph_str = string.replace(dep_graph_str, '\\n', '\n')
  #dep_graph_str = string.replace(dep_graph_str, '\\\'', '\'')
  lemma_str = line[3]
  words_str = line[2][1:-1]
  words_str = string.replace(words_str, "\",\"", "~^~")
  # skip sentences with empty dependency graphs
  #if dep_graph_str == "":
  #  return ""
  types = [line[9], line[13]]
  starts = [line[14], line[16]]
  ends = [line[15], line[17]]
  lemma = lemma_str.split(ARR_DELIM)
  dep_graph = dep_graph_str.split("\n")
  #PATTERN = re.compile(r'''((?:"[^"]*")+)''')
  #words = PATTERN.split(words_str[1:-1])[1::2]
  words = words_str.split(",")
  for i,word in enumerate(words):
	if word == "~^~":
		words[i] = ','
  mention_ids = [line[7], line[11]]
  mention_words = [[words[int(starts[0]): int(ends[0])]],[words[int(starts[1]):int(ends[1])]]]
  # create a list of mentions
  mentions = zip(mention_ids, mention_words, types, starts, ends)
  mentions = map(lambda x: {"mention_id" : x[0], "word" : x[1], "type" : x[2], "start" : int(x[3]), "end" : int(x[4])}, mentions)

  relation = None
  if len(line) == 21:
	relation = line[18]
  #now we get the path from both mentions to the root
  # get a list of Word object
  obj = {}
  obj['lemma'] = lemma
  obj['words'] = words
  obj['dep_graph'] = dep_graph
  word_obj_list = ddlib.unpack_words(obj, lemma='lemma', words='words', dep_graph='dep_graph', dep_graph_parser=dep_format_parser)

  m1 = mentions[0]
  m2 = mentions[1]
  #print row
  if m1 != m2:
	link, path = ddlib.dep_path_between_words_new(word_obj_list, int(ends[0])-1, int(ends[1])-1)
	feat = [m1["word"], m2["word"], m1["type"], m2["type"], path, link]
	if relation is not None:
		feat.append(relation)
	return feat

#outputs just the words along the dependency path
def get_basic_rnn_features(line):
	lis = get_recurrent_features_new(line)
	ret = []
	ret.extend(lis[0][0])
	ret.extend(lis[5][1:-1])
	ret.extend(lis[1][0])
	rel = lis[6]
	return ret, rel	

for line in sys.stdin:
	temp, rel =  get_basic_rnn_features(line)
 	if temp == None:
		break
	out = " ".join(temp)
	if rel is not None:
		rel = rel[1:-1]
		rel = rel.split(",")
		#rel = rel[np.random.randint(0, len(rel))]
		out = out + "\t" + rel[0]
	print out
        #print '-----------------------------------------------------------------------------------------------------------------------'
