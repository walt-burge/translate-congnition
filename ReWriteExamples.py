import argparse
import pandas
import os
import tensorflow as tf

DATA_DIR = '.'

TRAIN_DATA_FILE = 'X_train.csv'
TEST_DATA_FILE = 'X_test.csv'

DATA_FILES = [TRAIN_DATA_FILE]

NUM_VARS = ['is_root', 'is_first', 'is_last', 'prev_cat', 'pos_cat', 'next_cat', 's_seq', 'n_p_seq', 'parent_wt']
POS_VARS = [ 'grndprnt', 'prnt', 'pos', 'prv_pos', 'nxt_pos']
WORD_VARS = [ 'prev_word', 'next_word', 'word']

ALL_VARS = NUM_VARS + POS_VARS + WORD_VARS

is_root_ex = tf.constant(0, dtype=tf.int32)
is_first_ex = tf.constant(0, dtype=tf.int32)
is_last_ex = tf.constant(0, dtype=tf.int32)
prev_cat_ex = tf.constant(-1, dtype=tf.int32)
pos_cat_ex = tf.constant(-1, dtype=tf.int32)
next_cat_ex = tf.constant(-1, dtype=tf.int32)
sentence_seq_ex = tf.constant(0, dtype=tf.int32)
node_sentence_seq_ex = tf.constant(0., dtype=tf.float32)
parent_weight_ex = tf.constant(1, dtype=tf.int32)
grndprnt_ex = tf.constant("", dtype=tf.string)
prnt_ex = tf.constant("", dtype=tf.string)
pos_ex = tf.constant("", dtype=tf.string)
pos_ind_ex = tf.constant(-1, dtype=tf.int32)
prv_pos_ex = tf.constant("", dtype=tf.string)
nxt_pos_ex = tf.constant("", dtype=tf.string)
prv_wrd_ex = tf.constant("", dtype=tf.string)
nxt_wrd_ex = tf.constant("", dtype=tf.string)
wrd_ex = tf.constant("", dtype=tf.string)
seq_length_ex = tf.constant(0, dtype=tf.int32)


def create_file_reader_ops(filename_queue, record_defaults):
    reader = tf.TextLineReader(skip_header_lines=False)
    _, csv_row = reader.read(filename_queue)
    is_root, is_first, is_last, prev_cat, pos_cat, next_cat, s_seq, n_p_seq, parent_wt, \
    grndprnt, prnt, pos, prv_pos, nxt_pos, prv_wrd, nxt_wrd, word \
        = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = [is_root, is_first, is_last, prev_cat, pos_cat, next_cat, s_seq, n_p_seq, parent_wt,
                        grndprnt, prnt, pos, prv_pos, nxt_pos, prv_wrd, nxt_wrd, word]
    return features, word


def create_text_dataset(filepath, pos_index, word_index, record_defaults, perform_shuffle=False, repeat_count=1):
    def decode_ln(line):
        is_root, is_first, is_last, prev_cat, pos_cat, next_cat, s_seq, n_p_seq, parent_wt, \
        grndprnt, prnt, pos, prv_pos, nxt_pos, prv_wrd, nxt_wrd, word, seq_length =\
        tf.decode_csv(line, record_defaults=record_defaults)

        vars = ['pos','prnt','prv_pos', 'nxt_pos', 's_seq']
        features = [pos_index.lookup(pos),
                    pos_index.lookup(prnt),
                    pos_index.lookup(prv_pos),
                    pos_index.lookup(nxt_pos),
                    s_seq]
        example = dict(zip(vars, features))
        return example, n_p_seq

    dataset = tf.data.TextLineDataset(filepath).map(decode_ln)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=512)

    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(100)

    return dataset

def create_pos_feature_columns(vars):
    columns = [tf.feature_column.categorical_column_with_vocabulary_file(key=var, vocabulary_file=pos_vocab_file,
                                                                         vocabulary_size=500) for var in POS_VARS]
    return columns


def create_word_feature_columns(vars):
    columns = [tf.feature_column.categorical_column_with_vocabulary_file(key=var, vocabulary_file=word_vocab_file,
                                                                         vocabulary_size=10000) for var in WORD_VARS]
    return columns


pos_vocab_file = os.path.join(DATA_DIR, 'pos_vocab.txt')
word_vocab_file = os.path.join(DATA_DIR, 'word_vocab.txt')


def main():
    parser = argparse.ArgumentParser(description='Generate CRF features from selected Treebank.')
    parser.add_argument('--data', type=str, help='Folder containing subdirectories ./eng_news_txt_tbnk-ptb_revised/ and ./ctb5.1_preproc. These folders contain the English and Chinese treebanks as specified in README.md. Note that the Chinese Treebank must be preprocessed prior to use.')

    args = parser.parse_args()

    global DATA_DIR
    global WORD_VOCAB_PATH
    global POS_VOCAB_PATH
    global model

    DATA_DIR = args.data

    WORD_VOCAB_PATH = os.path.join(DATA_DIR, 'word_vocab.txt')
    POS_VOCAB_PATH = os.path.join(DATA_DIR, 'pos_vocab.txt')

    rows = 0
    #next_batch = create_text_dataset(TEST_DATA_FILE, pos_index, True)
    #numeric_columns = [tf.feature_column.numeric_column(k) for k in NUM_VARS]
    #pos_columns = create_pos_feature_columns(POS_VARS)
    #word_columns = create_word_feature_columns(WORD_VARS)

    word = tf.feature_column.categorical_column_with_vocabulary_file(key='wrd', vocabulary_file=WORD_VOCAB_PATH,
                                                                     vocabulary_size=500, num_oov_buckets=5)
    pos = tf.feature_column.categorical_column_with_vocabulary_file(
        key='pos', vocabulary_file=os.path.join(DATA_DIR, POS_VOCAB_PATH), vocabulary_size=500,
        num_oov_buckets=5)
    parent = tf.feature_column.categorical_column_with_vocabulary_file(key='prnt', vocabulary_file=POS_VOCAB_PATH,
                                                                       vocabulary_size=500, num_oov_buckets=5)
    prev_pos = tf.feature_column.categorical_column_with_vocabulary_file(key='prv_pos', vocabulary_file=POS_VOCAB_PATH,
                                                                         vocabulary_size=500, num_oov_buckets=5)
    next_pos = tf.feature_column.categorical_column_with_vocabulary_file(key='nxt_pos', vocabulary_file=POS_VOCAB_PATH,
                                                                         vocabulary_size=500, num_oov_buckets=5)
    s_seq = tf.feature_column.numeric_column('s_seq', dtype=tf.int32)

    record_defaults = [is_root_ex, is_first_ex, is_last_ex,
                       prev_cat_ex, pos_cat_ex, next_cat_ex,
                       sentence_seq_ex, node_sentence_seq_ex, parent_weight_ex,
                       grndprnt_ex, prnt_ex, pos_ex, prv_pos_ex, nxt_pos_ex, prv_wrd_ex, nxt_wrd_ex, wrd_ex,
                       seq_length_ex]

    smaller_defaults = [is_root_ex, is_first_ex, is_last_ex,
                        prev_cat_ex, pos_cat_ex, next_cat_ex,
                        sentence_seq_ex, node_sentence_seq_ex, parent_weight_ex,
                        pos_ind_ex, pos_ind_ex, pos_ind_ex, pos_ind_ex, pos_ind_ex, prv_wrd_ex, nxt_wrd_ex, wrd_ex]

    #feature_columns = [numeric_columns + pos_columns + word_columns]

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        pos_index = tf.contrib.lookup.index_table_from_file(POS_VOCAB_PATH)
        word_index = tf.contrib.lookup.index_table_from_file(WORD_VOCAB_PATH)

        tf.tables_initializer().run()

        dataset = create_text_dataset(os.path.join(DATA_DIR, TRAIN_DATA_FILE), pos_index, word_index, record_defaults)
        iterator = dataset.make_initializable_iterator('train_data')
        batch_features, batch_labels = iterator.get_next()

        sess.run([iterator.initializer])

        while True:
            try:
                print("Here's feature_data and label_data:")
                feature_data, label_data = sess.run([batch_features, batch_labels])
                print("Features:")
                print(feature_data)
                print("\nLabels:")
                print(label_data)

                print("And, here's example_data and word:")
                #example_data = sess.run(example)
                #print(example_data)
                #word = sess.run(word)
                #print(word)
                print("(not)")

                #print("Shape of feature data: {}".format(tf.shape(feature_data)))
                #print("Shape of label data: {}".format(tf.shape(label_data)))
                print("Low-tech shape of feature data: items: {}, keys: {}, values: {}".
                      format(len(feature_data.items()), len(feature_data.keys()), len(feature_data.values())))

                print("Low-tech shape of label data: {}".format(label_data.shape))
                #print("Shape of example data: {}".format(tf.shape(example)))
                #print("Shape of word data: {}".format(tf.shape(word)))

            except tf.errors.OutOfRangeError as oorEx:
                print("OutOfRangeError: code:{}, msg:{}, node:{}, op:{}".format(oorEx.error_code,
                                                                                oorEx.message,
                                                                                oorEx.node_def, oorEx.op))
                break
            rows += 1
            if rows > 0:
                print("{} rows. Now, that's a lot of data!".format(rows))
                break

    print("Total of {} rows observed".format(rows))

main()