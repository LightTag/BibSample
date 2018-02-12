'''
Tools to take a directory of txt files and convert them to TF records
'''
from collections import defaultdict, Counter
import numpy as np
import tensorflow as tf
PAD = "<PAD>"
EOS = "<EOS>"


class Preppy():
    '''
    Class that converts text inputs to numpy arrays of ids.
    It assigns ids sequentially to the token on the fly.
    '''
    def __init__(self, tokenizer_fn):
        self.vocab = defaultdict(self.next_value)
        self.word_count = Counter()
        self.vocab[PAD] = 0
        self.vocab[EOS] = 1
        self.next = 1
        self.tokenizer = tokenizer_fn
        self.reverse_vocab = {}

    def next_value(self):
        self.next += 1
        return self.next

    def sequence_to_tf_example(self,sequence):
        id_list = self.sentance_to_id_list(sequence)
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        sequence_length = len(sequence)
        ex.context.feature["length"].int64_list.value.append(sequence_length)
        # Feature lists for the two sequential features of our example
        fl_tokens = ex.feature_lists.feature_list["tokens"]

        for token in id_list:
            fl_tokens.feature.add().int64_list.value.append(token)

        return ex

    def convert_token_to_id(self, token):
        self.word_count[token] += 1
        return self.vocab[token]

    def sentance_to_tokens(self, sent):
        return self.tokenizer(sent)

    def tokens_to_id_list(self, tokens):
        return list(map(self.convert_token_to_id, tokens))

    def sentance_to_id_list(self, sent):
        tokens = self.sentance_to_tokens(sent)
        id_list = self.tokens_to_id_list(tokens)
        return id_list

    def sentance_to_numpy_array(self, sent):
        id_list = self.sentance_to_id_list(sent)
        return np.array(id_list)

    def update_reverse_vocab(self):
        self.reverse_vocab = {id_:token for token,id_ in self.vocab.items()}
    def id_list_to_text(self,id_list):
        tokens = ''.join(map(lambda x:self.reverse_vocab[x],id_list))
        return tokens
    @staticmethod
    def parse(ex):
        '''
        Explain to TF how to go froma  serialized example back to tensors
        :param ex:
        :return:
        '''
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"seq":sequence_parsed["tokens"], "length": context_parsed["length"]}


class BibPreppy(Preppy):
    '''
        We'll slightly extend to way we right tfrecords to store the id of the book it came from
        '''
    def __init__(self,tokenizer_fn):
        super(BibPreppy,self).__init__(tokenizer_fn)
        self.book_map ={}

    def sequence_to_tf_example(self, sequence, book_id):
        id_list = self.sentance_to_id_list(sequence)
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        sequence_length = len(sequence)
        ex.context.feature["length"].int64_list.value.append(sequence_length)
        ex.context.feature["book_id"].int64_list.value.append(book_id)
        # Feature lists for the two sequential features of our example
        fl_tokens = ex.feature_lists.feature_list["tokens"]

        for token in id_list:
            fl_tokens.feature.add().int64_list.value.append(token)

        return ex
    @staticmethod
    def parse(ex):
        '''
        Explain to TF how to go froma  serialized example back to tensors
        :param ex:
        :return:
        '''
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64),
            "book_id": tf.FixedLenFeature([], dtype=tf.int64)
        }
        sequence_features = {
            "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        }

        # Parse the example (returns a dictionary of tensors)
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return {"seq": sequence_parsed["tokens"], "length": context_parsed["length"],
                "book_id": context_parsed["book_id"]}



