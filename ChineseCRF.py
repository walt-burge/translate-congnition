#!/Users/burgew/Develop/py2_env/bin/python
# coding: utf-8

from ast import literal_eval as make_tuple
import codecs
import collections
import glob
import io
import nltk
from nltk.tag.util import untag
from nltk import ParentedTree
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from lxml import etree
import re
model = None

def features(tagged, sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],

        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


def my_features(word, tag, cat, previous_words, next_words):
    prev2_word = None
    prev1_word = None
    next1_word = None
    next2_word = None

    if len(previous_words) > 0:
        prev1_word = previous_words[0]
        if len(previous_words) > 1:
            prev2_word = previous_words[1]

    if len(next_words) > 0:
        next1_word = next_words[0]
        if len(next_words) > 1:
            next2_word = next_words[1]

    return {
        'word': word,
        'tag': tag,
        'cat': cat,
        'is_first': prev1_word is None,
        'is_last': len([]) == 0,
        'prev2_word_tag': prev2_word[1] if prev2_word is not None else '',
        'prev2_word_word': prev2_word[0] if prev2_word is not None else '',
        'prev1_word_tag': prev1_word[1] if prev1_word is not None else '',
        'prev1_word_word': prev1_word[0] if prev1_word is not None else '',
        'next2_word_tag': next2_word[1] if next2_word is not None else '',
        'next2_word_word': next2_word[0] if next2_word is not None else '',
        'next1_word_tag': next1_word[1] if next1_word is not None else '',
        'next1_word_word': next1_word[0] if next1_word is not None else '',
    }


def is_tree(node):
    return isinstance(node, nltk.tree.Tree)


def iterate_features(parsed):
    previous_words = []

    cat = None
    for node in parsed:
        word = node[0]

        if is_tree(word):
            cat = node.label()
        else:
            tag = node.label()

            previous_words, next_words = [], []

            if isinstance(node, ParentedTree):
                while node.left_sibling() is not None:
                    previous_words.append(node.left_subling())
                    if len(previous_words) > 1:
                        break

                while node.right_sibling() is not None:
                    next_words.append(node.right_subling())
                    if len(next_words) > 1:
                        break

            features = my_features(word, tag, cat, previous_words, next_words)
            yield features

            if is_tree(node):
                cat = None
                for child in node:
                    iterate_features(child)


def transform_to_dataset(parsed_sentences):
    X, y = [], []
 
    for parsed in parsed_sentences:
        #X.append([features(tagged, untag(tagged), index) for index in range(len(tagged))])
        nodes = [x for x in iterate_features(parsed)]
        if nodes is not None:
            for node in nodes:
                X.append(node)
                y.append(node["tag"])
            print("X length: {}".format(len(X)))
            print("y length: {}".format(len(y)))
 
    return X, y


def pos_tag(sentence):
    sentence_features = [iterate_features(sentence, index) for index in range(len(sentence))]

    return list(zip(sentence, model.predict([sentence_features])[0]))


def get_ctb_sentences(filepath):

    with io.open(filepath, encoding='ISO-8859-1') as file:
        in_sentence = False
        in_chinese = False
        chinese_text = ''

        sentence = ""
        for line in file:
            line = line.replace('\t','')
            line = line.replace('\n','')
            line = line.replace('  ', ' ')
            line = line.strip()
            if line.startswith('</S'):
                if in_sentence:
                    in_sentence = False
                    #sentence = re.sub(r"\)\s+\(", "),(", sentence)
                    a_list = [p.split() for p in re.findall('\w+\s+\d+', sentence)]
                    #line = OneOrMore(nestedExpr()).parseString(encoded_line)
                    string = codecs.unicode_escape_encode(sentence, 'ISO-8859-1')
                    yield nltk.Tree.fromstring(sentence)
            elif line.startswith('<S'):
                in_sentence = True
            elif in_sentence:

                #encoded_line = codecs.encode(line, 'ISO-8859-1')
                for char in line:
                    if ord(char) > 128:
                        if not in_chinese:
                            in_chinese = True
                        #char = codecs.encode(char, 'ISO-8859-1')
                        chinese_text += char
                    else:
                        if in_chinese:
                            sentence = sentence.join(chinese_text, ' ')
                            #sentence += codecs.decode(chinese_text, 'ISO-8859-1')
                            in_chinese = False
                        sentence += char

def get_ugly_problems():
    tagged_sentences = []

    for bracketed_filepath in glob.glob('/Users/burgew/Pre-Trained/ctb5.1_preproc/*.txt'):
        with io.open(bracketed_filepath, encoding='ISO-8859-1') as file:
            tagged_sentences.append(ParentedTree.fromstring(file.read()))
        #for tuple in get_ctb_sentences(bracketed_filepath):
        #    print tuple
        #    tagged_sentences.append(tuple)


def main():
    global model

    parsed_sentences = nltk.corpus.sinica_treebank.parsed_sents()[:50]
    tagged_words = nltk.corpus.sinica_treebank.tagged_words()[:100]

    print(parsed_sentences[0])
    print("Parsed sentences: ", len(parsed_sentences))
    print("Tagged words: ", len(tagged_words))

    # Split the dataset for training and testing
    cutoff = int(.75 * len(parsed_sentences))
    training_sentences = parsed_sentences[:cutoff]
    test_sentences = parsed_sentences[cutoff:]
    X_train, y_train = transform_to_dataset(training_sentences)
    X_test, y_test = transform_to_dataset(test_sentences)

    print(len(X_train))
    print(len(X_test))
    print(X_train[0])
    print(y_train[0])

    model = CRF()
    model.verbose = True
    model.fit(X_train, y_train)

    sentence = [char for char in '表演基本上很精彩--我只对她的技巧稍有意见']

    print(pos_tag(sentence))

    y_pred = model.predict(X_test)
    print(metrics.flat_accuracy_score(y_test, y_pred))


main()



