#!/Users/burgew/Develop/py2_env/bin/python
# coding: utf-8

import argparse
import csv
import glob
import io
import os
import json
import nltk
from nltk import ParentedTree

DATA_DIR = '/Users/burgew/Pre-Trained/translate-cognition'

NUMERIC_COLUMNS = [
        'root',
        'is_first',
        'is_last',
        'prev_cat',
        'pos_cat',
        'next_cat',
        's_sequence',
        'n_p_sequence',
        'parent_weight']

POS_COLUMNS = [ 'grandparent',
                'parent',
                'pos',
                'prev_pos',
                'next_pos']

WORD_COLUMNS = [
    'prev_word',
    'next_word',
    'word']

CATEGORICAL_COLUMNS = POS_COLUMNS
CATEGORICAL_COLUMNS.extend(WORD_COLUMNS)

model = None

word_vocab = set()
pos_vocab = set()
nonterm_vocab = set()

tb_map = None

def read_tb_map(tb_map_filepath):
    print("Reading {}".format(tb_map_filepath))
    tb_map_dict = {}
    with open(tb_map_filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print("Column names are {}, {}, {}".format(row[0], row[1], row[2]))
                line_count += 1
            else:
                print("\tPOS {} has description {}, and POS category (id) {}.".format(row[1], row[2], row[0]))
                line_count += 1
                tb_map_dict[row[1]] = row[0]
        print('Processed {} lines'.format(line_count))
        return tb_map_dict

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1


def setup_tb_map(tb_type):
    global tb_map_dict

    if tb_type == "ctb":
        tb_map_dict = read_tb_map("./TB.POS.Map/CTB-map.csv")
    elif tb_type == "ptb":
        tb_map_dict = read_tb_map("./TB.POS.Map/PTB-map.csv")
    else:
        raise ValueError("Error, tb_type must be either 'ptb' or 'ctb'")

    return tb_map_dict


def get_tb_map(tb_type=None):
    global tb_map_dict

    if (tb_map is None):
        tb_map_dict = setup_tb_map(tb_type)
    elif tb_type is None:
        raise ValueError("Error, TB POS Map not loaded.")

    return tb_map_dict




class NodeFeatureParser():
    # Parser to generate features from a sentence dependency tree
    #     tb_type - either "ptb" or "ctb"

    flat_sentence = []

    def __init__(self, node, sequence=0, ancestors=None, root=False,
                 parser=None, tb_type=None, tb_map=None):
        self.node = node
        self.sequence = sequence
        self.ancestors = ancestors if ancestors is not None else []
        self.root = root

        if parser is not None:
            # Inherit context from parent parser instance
            self.tb_map = parser.tb_map

            if (self.node.label() == 'S') or (self.node.label() == 'ROOT') or (self.node.label() == ''):
                # For each sentence, get a flattened set of POS nodes at the word level
                self.flat_sentence = self.node.pos()
            else:
                self.flat_sentence = parser.flat_sentence
                self.sequence_length = parser.sequence_length

        else:
            if tb_map is not None:
                self.tb_map = tb_map
            else:
                # Otherwise, need to get a tb_map as requested, along with creating a flat representation of the sentence
                if (tb_type is None) or (tb_type != 'ptb' and tb_type !='ctb'):
                    raise ValueError("Error, node_feature_parser requires tb_type parameter = 'ptb' or 'ctb'")
                else:
                    self.tb_map = get_tb_map(tb_type)
            if (self.node.label() == 'S') or (self.node.label() == 'ROOT') or (self.node.label() == ''):
                # For each sentence, get a flattened set of POS nodes at the word level
                self.flat_sentence = self.node.pos()


        self.sequence_length = len(self.flat_sentence)
        # Derive a node mass from the flattening of the tree below this node, or just 1 is this is a word node
        self.node_mass = len(self.node.flatten()) if self.is_tree() else 1

    def is_tree(self):
        return isinstance(self.node[0], ParentedTree)

    def get_cat(self, pos):
        cat = self.tb_map.get(pos)

        return int(cat) if cat is not None else -1

    def get_word_features(self):
        flat_self = self.node.flatten()
        return {
            'word': flat_self[0],
            'pos': flat_self[0].label()
        }

    def add_sibling_features(self, current_features):
        #print("len(self.node): {}, self.sequence: {}, len(self.flat_sentence): {}, self.flat_sentence: {}".format(len(self.node), self.sequence, len(self.flat_sentence), self.flat_sentence))
        if (self.sequence > 1) and (len(self.flat_sentence)>self.sequence):
            prev_word = self.flat_sentence[self.sequence - 1][0]
            prev_pos = self.flat_sentence[self.sequence - 1][1]
            prev_cat = self.get_cat(prev_pos)
            current_features["prev_word"] = prev_word
            current_features["prev_pos"] = prev_pos
            current_features["prev_cat"] = prev_cat
            current_features["is_first"] = 0
        if self.sequence < (len(self.flat_sentence)-1):
            next_word = self.flat_sentence[self.sequence + 1][0]
            next_pos = self.flat_sentence[self.sequence + 1][1]
            next_cat = self.get_cat(next_pos)
            current_features["next_word"] = next_word
            current_features["next_pos"] = next_pos
            current_features["next_cat"] = next_cat
            current_features["is_last"] = 0

    def get_features(self, word, pos, root=False):
        parent = self.ancestors[-1]

        parent_sequence = int(parent[1][0])
        parent_weight = int(parent[1][1])

        if len(self.ancestors) > 1:
            grandparent = self.ancestors[-2]

            grandparent_sequence = float(grandparent[1][0])
            grandparent_weight = int(grandparent[1][1])

            # This divides the parent's positive/negative relative sequence under the grandparent
            # weightequence by the grandparent's mass, giving the relative sequencing within the grandparent
            parent_rel_sequence = (parent_sequence - grandparent_weight / 2) / grandparent_weight

            # This divides the parent's mass by the grandparent's mass, giving the relative importance within the grandparent
            parent_rel_weight = (grandparent_weight / 2 - parent_weight) / grandparent_weight

        elif len(self.ancestors) > 0:
            grandparent = None
            parent_weight = 1
            parent_sequence = 1

        else:
            grandparent = None
            parent = None
            parent_weight = 1
            parent_sequence = 1

        if parent_weight == 1:
            node_prel_sequence = 1
        else:
            # Node sequence gives the positive/negative (before or after midpoint) in parent
            unnorm_prel = (self.sequence - parent_weight / 2) / parent_weight
            node_prel_sequence = 0 if unnorm_prel == 0 else unnorm_prel/abs(unnorm_prel)

        return {
            # numeric features 0-8
            'root': 1 if root else 0,
            'is_first': 1,
            'is_last': 1,
            'prev_cat': -1,
            'pos_cat': self.get_cat(pos),
            'next_cat': -1,
            's_sequence': self.sequence,  # This is the overall word sequence (0-1) within sentence
            'n_p_sequence': node_prel_sequence,
            'parent_weight': parent_weight,
            # string features 9-16
            'grandparent': self.ancestors[-2][0] if len(self.ancestors) > 1 else None,
            'parent': self.ancestors[-1][0] if len(self.ancestors) > 0 else None,
            'pos': pos,
            'prev_pos': '',
            'next_pos': '',
            'prev_word': '',
            'next_word': '',
            # word is the example identity
            'word': word
        }

    def iterate_features(self):

        subnode_features = []

        if self.is_tree(): # This is a sub-tree node
            subnode_ancestors = self.ancestors
            subnode_ancestors.append((self.node.label(), (self.sequence, self.node_mass)))

            for ind, p_subtree in enumerate(self.node[0:len(self.node)]):

                p_subtree_parser = NodeFeatureParser(node=p_subtree, sequence=self.sequence+ind,
                                                     ancestors=subnode_ancestors, parser=self)
                for sub_features in p_subtree_parser.iterate_features():
                    subnode_features.append(sub_features)

                # Note that, after parsing the first node under the 'S', we're no longer at the root
                if ind>0:
                    root=False

        else: # This is a word node
            these_features = self.get_features(word=self.node[0], pos=self.node.label())
            if (len(self.node)>0 and len(self.flat_sentence)>0) and \
                    ((self.sequence>0) or (self.sequence<len(self.flat_sentence))):
                self.add_sibling_features(these_features)

            subnode_features.append(these_features)

        for node_features in subnode_features:
            yield(node_features)

