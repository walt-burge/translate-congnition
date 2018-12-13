import argparse
import numpy as np
import os
import tensorflow as tf

DATA_DIR = '/Users/burgew/Pre-Trained/translate-cognition/'

TRAIN_DATA_FILE = 'X_train.csv'
TEST_DATA_FILE = 'X_test.csv'

# Data settings.
batch_size = 100
num_examples = 4000
num_words = 20
num_features = 5
num_tags = 3

shuffle = False
repeat_count = 1

# Random features.
x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)

# Random tag indices representing the gold sequence.
y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.int32)

# All sequences in this example have the same length, but they can be variable in a real model.
sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)

NUM_VARS = ['is_root', 'is_first', 'is_last', 'prev_cat', 'pos_cat', 'next_cat', 's_seq', 'n_p_seq', 'parent_wt']
POS_VARS = [ 'grndprnt', 'prnt', 'pos', 'prv_pos', 'nxt_pos']
WORD_VARS = [ 'prev_word', 'next_word', 'word']

ALL_VARS = NUM_VARS + POS_VARS + WORD_VARS




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

    # Train and evaluate the model.
    with tf.Graph().as_default():
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

        record_defaults = [is_root_ex, is_first_ex, is_last_ex,
                           prev_cat_ex, pos_cat_ex, next_cat_ex,
                           sentence_seq_ex, node_sentence_seq_ex, parent_weight_ex,
                           grndprnt_ex, prnt_ex, pos_ex, prv_pos_ex, nxt_pos_ex, prv_wrd_ex, nxt_wrd_ex, wrd_ex,
                           seq_length_ex]

        sentence_length_ex = tf.constant(0, dtype=tf.int32)
        seq_length_feature = tf.feature_column.numeric_column('seq_length', dtype=tf.int32)

        with tf.Session() as session:
            pos_index = tf.contrib.lookup.index_table_from_file(POS_VOCAB_PATH)
            word_index = tf.contrib.lookup.index_table_from_file(WORD_VOCAB_PATH)

            train_csv_path = os.path.join(DATA_DIR, TRAIN_DATA_FILE)

            word = tf.feature_column.categorical_column_with_vocabulary_file(key='wrd',
                                                                             vocabulary_file='./word_vocab.txt',
                                                                             vocabulary_size=500, num_oov_buckets=5)
            pos = tf.feature_column.categorical_column_with_vocabulary_file(
                key='pos', vocabulary_file=POS_VOCAB_PATH, vocabulary_size=500,
                num_oov_buckets=5)
            parent = tf.feature_column.categorical_column_with_vocabulary_file(key='prnt',
                                                                               vocabulary_file=POS_VOCAB_PATH,
                                                                               vocabulary_size=500, num_oov_buckets=5)
            prev_pos = tf.feature_column.categorical_column_with_vocabulary_file(key='prv_pos',
                                                                                 vocabulary_file=POS_VOCAB_PATH,
                                                                                 vocabulary_size=500, num_oov_buckets=5)
            next_pos = tf.feature_column.categorical_column_with_vocabulary_file(key='nxt_pos',
                                                                                 vocabulary_file=POS_VOCAB_PATH,
                                                                                 vocabulary_size=500, num_oov_buckets=5)
            s_seq = tf.feature_column.numeric_column('s_seq', dtype=tf.int32)

            features = [pos, parent, prev_pos, next_pos, s_seq]

            def decode_ln(line):
                is_root, is_first, is_last, prev_cat, pos_cat, next_cat, s_seq, n_p_seq, parent_wt, \
                grndprnt, prnt, pos, prv_pos, nxt_pos, prv_wrd, nxt_wrd, word, seq_length = \
                    tf.decode_csv(line, record_defaults=record_defaults)

                vars = ['pos', 'prnt', 'prv_pos', 'nxt_pos', 's_seq']
                features = [pos_index.lookup(pos),
                            pos_index.lookup(prnt),
                            pos_index.lookup(prv_pos),
                            pos_index.lookup(nxt_pos),
                            s_seq]
                example = dict(zip(vars, features))
                return example, n_p_seq



            dataset = tf.data.TextLineDataset(train_csv_path).map(decode_ln)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=512)

            dataset = dataset.repeat(repeat_count)
            dataset = dataset.batch(100)

            #dataset = create_text_dataset(os.path.join(DATA_DIR, TRAIN_DATA_FILE), pos_index, word_index, record_defaults)
            iterator = dataset.make_initializable_iterator('train_data')
            batch_features, batch_labels = iterator.get_next()

            session.run([iterator.initializer])

            # Add the data to the TensorFlow graph.

            features_t = tf.placeholder(tf.int32, shape=(batch_size, num_features))
            seq_length_t = tf.placeholder(tf.int32, shape=(batch_size))
            x_t = tf.constant(features)
            y_t = tf.constant(batch_labels)

            # Compute unary scores from a linear layer.
            weights = tf.get_variable("weights", [num_features, num_tags])
            matricized_x_t = tf.reshape(batch_features, [-1, num_features])
            matricized_unary_scores = tf.matmul(matricized_x_t, weights)
            unary_scores = tf.reshape(matricized_unary_scores,
                                      [num_examples, num_words, num_tags])

            # Compute the log-likelihood of the gold sequences and keep the transition
            # params for inference at test time.
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                unary_scores, y_t, seq_length_t)

            # Compute the viterbi sequence and score.
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
                unary_scores, transition_params, seq_length_t)

            # Add a training op to tune the parameters.
            loss = tf.reduce_mean(-log_likelihood)
            train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

            session.run(tf.global_variables_initializer(),
                        feed_dict={
                            features_t : features,
                            seq_length_t : seq_length_feature
                        })

            mask = (np.expand_dims(np.arange(num_words), axis=0) <
                    np.expand_dims(sequence_lengths, axis=1))
            total_labels = np.sum(sequence_lengths)

            # Train for a fixed number of iterations.
            for i in range(1000):
                tf_viterbi_sequence, _ = session.run([viterbi_sequence, train_op])
                if i % 100 == 0:
                    correct_labels = np.sum((y == tf_viterbi_sequence) * mask)
                    accuracy = 100.0 * correct_labels / float(total_labels)
                    print("Accuracy: %.2f%%" % accuracy)

main()