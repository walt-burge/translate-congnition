import argparse
import csv
import io
import os
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = '.'

TRAIN_DATA_FILE = 'X_train.csv'
TEST_DATA_FILE = 'X_test.csv'

NUM_DATA_LABELS = ['is_root', 'is_first', 'is_last',
                   'prev_cat', 'pos_cat', 'next_cat',
                   's_sequence', 'n_p_sequence', 'parent_weight']
STR_DATA_LABELS= ['grandparent', 'parent', 'word', 'pos',
                  'prev_pos', 'prev_word', 'next_pos', 'next_word ']

tf.enable_eager_execution()

def convert_num_to_tf_file(data_dir, num_input_file, data_set='train'):
    input_filename = os.path.join(data_dir, num_input_file)
    output_filename = os.path.join(data_dir, 'num_data.' + data_set + '.tfrecords')
    print('Writing', output_filename)
    with tf.python_io.TFRecordWriter(output_filename) as writer:
        with io.open(num_input_file, "r", encoding='ISO-8859-1') as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                is_root, is_first, is_last  = int(arr[0]), int(arr[1]), int(arr[2])
                prev_cat, pos_cat, next_cat = int(arr[3]), int(arr[4]), int(arr[5])
                s_seq, n_p_seq, par_wt = float(arr[6]), float(arr[7]), float(arr[8])
                is_root_arr = np.reshape(is_root,[1]).astype('int32')
                is_first_arr = np.reshape(is_first,[1]).astype('int32')
                is_last_arr = np.reshape(is_last,[1]).astype('int32')
                prev_cat_arr = np.reshape(prev_cat,[1]).astype('int32')
                pos_cat_arr = np.reshape(pos_cat,[1]).astype('int32')
                next_cat_arr = np.reshape(next_cat,[1]).astype('int32')
                s_seq_arr = np.reshape(s_seq,[1]).astype('float32')
                n_p_seq_arr = np.reshape(n_p_seq,[1]).astype('float32')
                par_wt_arr = np.reshape(par_wt,[1]).astype('float32')
                example = tf.train.Example()
                example.features.feature["is_root"].int64_list.value.extend(is_root_arr)
                example.features.feature["is_first"].int64_list.value.extend(is_first_arr)
                example.features.feature["is_last"].int64_list.value.append(is_last_arr)
                example.features.feature["prev_cat"].int64_list.value.extend(prev_cat_arr)
                example.features.feature["pos_cat"].int64_list.value.extend(pos_cat_arr)
                example.features.feature["next_cat"].int64_list.value.append(next_cat_arr)
                example.features.feature["s_sequence"].int64_list.value.append(s_seq_arr)
                example.features.feature["n_p_sequence"].float32_list.value.append(n_p_seq_arr)
                example.features.feature["p_weight"].float32_list.value.append(par_wt_arr)
                writer.write(example.SerializeToString())
                line = f.readline()

def main():
    parser = argparse.ArgumentParser(description='Generate CRF features from selected Treebank.')
    parser.add_argument('--data', type=str,
                        help='Folder containing subdirectories ./eng_news_txt_tbnk-ptb_revised/ and ./ctb5.1_preproc. These folders contain the English and Chinese treebanks as specified in README.md. Note that the Chinese Treebank must be preprocessed prior to use.')

    args = parser.parse_args()

    global DATA_DIR

    DATA_DIR = args.data

    convert_num_to_tf_file(DATA_DIR, TRAIN_DATA_FILE, data_set='train')
    convert_num_to_tf_file(DATA_DIR, TEST_DATA_FILE, data_set='test')
    
    with tf.Session() as sess:
        tf.enable_eager_execution()
        sess.run(tf.global_variables_initializer())

        is_root = tf.placeholder(tf.int32)
        is_first = tf.placeholder(tf.int32)
        is_last = tf.placeholder(tf.int32)
        prev_cat = tf.placeholder(tf.int32)
        pos_cat = tf.placeholder(tf.int32)
        next_cat = tf.placeholder(tf.int32)
        s_sequence = tf.placeholder(tf.float32)
        n_p_sequence = tf.placeholder(tf.float32)
        parent_weight = tf.placeholder(tf.float32)


        num_dataset = tf.data.experimental.make_csv_dataset(
           TEST_DATA_FILE,
            batch_size=20,
            num_epochs=1,
            column_names=NUM_DATA_LABELS,
            column_defaults=[is_root, is_first, is_last,
                             prev_cat, pos_cat, next_cat,
                             s_sequence, n_p_sequence, parent_weight],
            header=False)


        str_dataset = tf.data.experimental.make_csv_dataset(
            TEST_DATA_FILE,
            batch_size=20,
            num_epochs=1,
            column_names=STR_DATA_LABELS,
            #column_defaults=[tf.string, tf.string, tf.string, tf.string,
            #                 tf.string, tf.string, tf.string, tf.string],
            header=False)


        sess.run(num_dataset, tf.shape(num_dataset))


main()