import random as rand
from random import random
import numpy as np
import math

# Parameters used.
MODEL_PATH = 'model/model.ckpt'

word_ind_map = {}


def is_word_head(word):
    return word[0].isupper()


# Return an indicator of whether the word is a Head
def is_word_ind_head(word_ind):
    return word_ind[1]


# Get a word index from the map
def get_word_ind(word):
    if word.lower() not in word_ind_map.keys():
        word_ind_map[word.lower()] = (len(word_ind_map.values()) + 1, is_word_head(word))

    return word_ind_map[word.lower()]


def classify(w_before, word, w_after):
    previous_root = False
    next_root = False

    classification = 0

    if not np.isnan(w_before[0]) and w_before[1]:
        classification = 1
    elif not np.isnan(w_after[0]) and w_after[1]:
        classification = -1

    return classification


def get_chunk(words, index):

    parent_visible = False
    chunk = []
    chunk_ind = -1

    # If the word in question is not root, find and return the chunk that includes the word
    if not words[index][1]:
        for ind, word in enumerate(words):
            # If a head word is encountered, reset the chunk to start with it
            if word[1]:
                parent_visible = True
                chunk = []
            if parent_visible:
                chunk.append(word)
                if ind == index and parent_visible:
                    chunk_ind = len(chunk)-1

            if parent_visible and len(chunk)>1 and is_word_ind_head(chunk[-1]):
                chunk = [word for word in chunk[0:-2]]
                break

    return chunk, chunk_ind


def classify_window(words, index):

    print("classify_window(")
    msg = ""
    for word in words:
        msg = msg + str(word)
    print(msg)
    print("")
    print("Index: {}".format(index))


    w_before = (np.nan, np.nan)
    w_after = (np.nan, np.nan)
    classification = 1

    word = words[index]

    chunk, chunk_ind = get_chunk(words, index)

    print("classify_window() : chunk({}), chunk_ind: {}".format(len(chunk), chunk_ind))

    if len(chunk) > 0:
        if chunk_ind > 0:
            w_before = chunk[chunk_ind - 1]

        if chunk_ind < len(chunk) - 1:
            w_after = chunk[chunk_ind + 1]

        classification = classify(w_before, word, w_after)
    return classification


# Create a sequence classification instance.
def get_sequence(sentence):

    print("get_sequence({})".format(sentence))
    # Create a sequence of random numbers in [0,1].
    X = [get_word_ind(word) for word in sentence.split()]

    # Determine the class outcome for each item in cumulative sequence.
    y = np.array([classify_window(X, ind) for ind, word in enumerate(X)])

    return X, y


# Create n examples with random sequence lengths between 5 and 15.
def get_examples():
    example_sentences = [
        "This is a test of the emergency broadcasting system.",
        "This is only a test.",
        "Had this been a real emergency, you would have received instructions.",
        "Test is just a Word you can use to Mean something big.",
        "Bigger, that is, and more Important."
    ]
    X_list = []
    y_list = []
    for sentence in example_sentences:
        X, y = get_sequence(sentence)
        X_list.append(X)
        y_list.append(y)

    return X_list, y_list


# Tensorflow requires that all sentences (and all labels) inside the same batch have the same length,
# so we have to pad the data (and labels) inside the batches (with 0's, for example).
def pad(sentence, max_length):
    pad_len = max_length - len(sentence)
    padding = np.zeros(pad_len)
    return np.concatenate((sentence, padding))


# Create input batches.
def batch(data, labels, sequence_lengths, batch_size, input_size):
    n_batch = int(math.ceil(len(data) / batch_size))
    index = 0
    for _ in range(n_batch):
        batch_sequence_lengths = np.array(sequence_lengths[index: index + batch_size])
        batch_length = np.array(max(batch_sequence_lengths))  # max length in batch
        batch_data = np.array([pad(x, batch_length) for x in data[index: index + batch_size]])  # pad data
        batch_labels = np.array([pad(x, batch_length) for x in labels[index: index + batch_size]])  # pad labels
        index += batch_size

        # Reshape input data to be suitable for LSTMs.
        batch_data = batch_data.reshape(-1, batch_length, input_size)

        yield batch_data, batch_labels, batch_length, batch_sequence_lengths

X, y = get_examples()

for ind, input in enumerate(X):
    print("Sentence: {}, classification: {}".format(input, y[ind]))