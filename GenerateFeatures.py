#!/Users/burgew/Develop/py2_env/bin/python
# coding: utf-8

import argparse
from ast import literal_eval as make_tuple
import codecs
import collections
import csv
import glob
import io
import os
import json
import nltk
from nltk import ParentedTree
from node_feature_parser import NodeFeatureParser, setup_tb_map

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
tb_type = None


def harvest_vocab_items(vocab_set, items):
    for item in items:
        if item not in vocab_set:
            vocab_set.add(item)


def save_vocab(vocab_set, filename):
    with open(os.path.join(DATA_DIR, filename), 'w') as vocab_file:
        for item in vocab_set:
            if len(item.strip()) > 0:
                vocab_file.write(item.strip()+'\n')


def adapt_json_data(dict_item, sequence_length):
    data_lst = [
    # numeric features 1-9
                 dict_item['root'],
                 dict_item['is_first'],
                 dict_item['is_last'],
                 dict_item['prev_cat'],
                 dict_item['pos_cat'],
                 dict_item['next_cat'],
                 dict_item['s_sequence'],
                 dict_item['n_p_sequence'],
                 dict_item['parent_weight'],
    # string features 10-16
                 dict_item['grandparent'],
                 dict_item['parent'],
                 dict_item['pos'],
                 dict_item['prev_pos'],
                 dict_item['next_pos'],
                 dict_item['prev_word'],
                 dict_item['next_word'],
                 dict_item['word'],
        # sentence length
                 sequence_length
               ]

    harvest_vocab_items(pos_vocab,[dict_item['pos'], dict_item['prev_pos'], dict_item['next_pos'],
                                   dict_item['grandparent'], dict_item['parent']])
    harvest_vocab_items(word_vocab,[dict_item['word'], dict_item['prev_word'], dict_item['next_word']])

    return data_lst


def feature_complete(features):
    return True;
    for value in features.values():
        if value is None:
            return False;


#def generate_features(parsed_sentence):
#    feature_parser = NodeFeatureParser(parsed_sentence, sequence=0, flat_sentence=[])
#    for features in feature_parser.iterate_features():
#        yield(features)


def transform_to_datasets(parsed_sentences, cutoff, tb_map):
    train_X_num, test_X_num = [], []

    X, y = [], []

    processed_sentences = 0

    for parsed in parsed_sentences:
        # Note that sequence and flat_sentence are mandatory only to ensure that they are specified for instances
        # The actual sequence is initiated by the S node and the interation of subtrees
        # The flat sentence is also initiated by the S node, by getting the flatten() at that node

        for feature_parser in [NodeFeatureParser(node=parsed, tb_map=tb_map) for x in parsed]:
            sequence_length = feature_parser.sequence_length
            for features in feature_parser.iterate_features():
                if processed_sentences <= cutoff:
                    train_X_num.append(adapt_json_data(features, sequence_length))
                else:
                    test_X_num.append(adapt_json_data(features, sequence_length))
            processed_sentences += 1

    print("train_X_num length: {}".format(len(train_X_num)))
    print("test_X_num length: {}".format(len(test_X_num)))

    return train_X_num, test_X_num


#def pos_pos(sentence):
#    sentence_features = [iterate_features(sentence, index) for index in range(len(sentence))]

#    return list(zip(sentence, model.predict([sentence_features])[0]))

def get_tb_sentence_count(folderpath, filename):
    sentence_count = 0

    for filepath in glob.glob(folderpath+"/"+filename):
        with io.open(filepath, encoding='ISO-8859-1') as file:
            sentence_count += sum(1 for line in file)

    return sentence_count


def get_treebank_sentences(folderpath, filename, sentence_count = -1):
    """

    :rtype: nltk.ParentedTree
    """
    for filepath in glob.glob(folderpath+"/"+filename):
        with io.open(filepath, encoding='ISO-8859-1') as file:
            retrieved_count = 0

            for line in file:
                line = line.replace('\t','')
                line = line.replace('\n','')
                line = line.replace('  ', ' ')
                line = line.strip()
                tree = nltk.ParentedTree.fromstring(line.strip())
                parented_tree = ParentedTree.convert(tree)
                yield parented_tree

                retrieved_count += 1
                if (sentence_count != -1) and (retrieved_count == sentence_count):
                    break


def main():
    global tb_type

    macbook_ctb_path = '/Users/burgew/Pre-Trained/ctb5.1_preproc'
    ubuntu_ctb_path = '/home/burgew/preprocessed/ctb_preproc'
    macbook_ptb_token_path = '/Users/burgew/Pre-Trained/eng_news_txt_tbnk-ptb_revised/data/tokenized_source/*/'
    macbook_ptb_path = '/Users/burgew/Pre-Trained/eng_news_txt_tbnk-ptb_revised/data/penntree/*/'

    parser = argparse.ArgumentParser(description='Generate CRF features from selected Treebank.')
    parser.add_argument('tb', type=str, choices=['ctb','ptb'],
                        help='Treebank for feature generation ("ctb" for Chinese Treebank or "ptb" for Penn Treebank')



    args = parser.parse_args()
    tb_type = args.tb

    tb_map = setup_tb_map(tb_type)

    global model

    if tb_type == "ptb":
        sentence_count = get_tb_sentence_count(macbook_ptb_token_path, '*1.txt')
        parsed_sentences = [sentence for sentence in get_treebank_sentences(macbook_ptb_path, '*1.tree')]
    else:
        sentence_count = get_tb_sentence_count(macbook_ctb_path, '*1.txt')
        parsed_sentences = [sentence for sentence in get_treebank_sentences(macbook_ctb_path, '*1.txt')]

    print(parsed_sentences[0:5])
    print("Parsed sentences: ", len(parsed_sentences))

    # Split the dataset for training and testing
    cutoff = int(.75 * sentence_count)
    X_train, X_test \
        = transform_to_datasets(parsed_sentences, cutoff, tb_map)

    with open('X_train.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(X_train)
    with open('X_test.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(X_test)

    for x in X_test[0:100]:
        print("{}:{}, {}:{}, {}:{}, {}:{}, {}:{}, {}:{}, {}:{}, {}:{}, {}:{}, {}:{} {}:{}, {}:{}, {}:{}, {}:{}, {}:{}, {}:{}, {}:{}, {}:{}".
              format("root", x[0],"frst", x[1],"last",x[2],"prv_cat",x[3],"pos_cat",x[4],"nxt_cat",x[5],
                     "s_seq", x[6], "n_p_seq", x[7], "prnt_wt", x[8],
                     "grnd", x[9], "prnt", x[10], "pos",x[11], "prv_pos", x[12], "nxt_pos", x[13],
                     "prv_wrd", x[14], "nxt_wrd", x[15], "word", x[16], "seq_length", x[17]))

    print("Size of X_train:")
    print(len(X_train))
    print("Size of X_test:")
    print(len(X_test))

    save_vocab(word_vocab, 'word_vocab.txt')
    print("Saved word vocab with {} words.".format(len(word_vocab)))
    save_vocab(pos_vocab, 'pos_vocab.txt')
    print("Saved pos vocab with {} unique pos.".format(len(pos_vocab)))

    print("")

    #model = CRF():1

    #model.verbose = True
    #model.fit(X_train, y_train)

    sentence = [char for char in '表演基本上很精彩--我只对她的技巧稍有意见']

    #print(pos_pos(sentence))

    #y_pred = model.predict(X_test)
    #print(metrics.flat_accuracy_score(y_test, y_pred))


main()



