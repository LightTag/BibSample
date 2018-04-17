import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import pickle

BP = pickle.load(open('./preppy.pkl', 'rb'))


class Model():
    def __init__(self, inputs, gs):
        sequence = inputs['seq']
        lengths = inputs['length']
        book_id = inputs['book_id']
        self.lr = tf.placeholder(shape=None, dtype=tf.float32)
        self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32)
        self.gs = gs
        self.increment_gs = tf.assign(self.gs, self.gs + 1)  # To increment during val

        with tf.variable_scope("main", initializer=xavier_initializer()):
            outputs, state = self.emebd_and_run_rnn(lengths, sequence)

            book_logits = self.get_book_logits(state)

            book_loss = tf.losses.sparse_softmax_cross_entropy(book_id, book_logits)

            lm_loss, lm_preds = self.get_language_model_loss(lengths, outputs, sequence)

            self.lm_preds = lm_preds
            self.total_loss = tf.reduce_mean(book_loss) + tf.reduce_mean(lm_loss)
            opt = tf.train.AdamOptimizer(self.lr)
            self.train = opt.minimize(self.total_loss, global_step=self.gs)
            self.target = book_id
            self.sequence = sequence
            self.lengths = lengths

            self.make_summaries(book_logits, book_loss, lm_loss)

    def make_summaries(self, book_logits, book_loss, lm_loss):
        '''
        Some summaries for Tensorflow
        '''
        self.pred = tf.argmax(tf.nn.softmax(book_logits), axis=1)
        self.accuracy = tf.contrib.metrics.accuracy(predictions=self.pred, labels=self.target)
        tf.summary.scalar("batch_accuracy", self.accuracy)
        tf.summary.scalar("total loss", self.total_loss)
        tf.summary.scalar("lm loss", lm_loss)
        tf.summary.scalar("book loss", book_loss)
        tf.summary.scalar("lr", self.lr)
        tf.summary.scalar("keep_prob", self.keep_prob)
        self.write_op = tf.summary.merge_all()

    @staticmethod
    def get_language_model_loss(lengths, outputs, sequence):
        '''
        We want the model to learn to predict the next character.
        We shift the input sequence one right and trim the right end of the outputs.
        So if sequence was
        ABCED ==> BCED
        And outputs (the GRU outputs)
        12345 ==> 1234

        Where we'd like to have the model learn that 1=B,2=C etc
        '''
        # Tensorflow supports Tensor slicing. Boom
        lm_target = sequence[:, 1:]
        # Even with negative indices. Boom Boom
        outputs = outputs[:, :-1, :]
        lm_logits = tf.contrib.layers.fully_connected(outputs, num_outputs=len(BP.vocab), activation_fn=None)
        mask = tf.sequence_mask(lengths - 1, tf.reduce_max(lengths) - 1)
        lm_preds = tf.argmax(tf.nn.softmax(lm_logits), axis=2)
        lm_loss = tf.losses.sparse_softmax_cross_entropy(lm_target, lm_logits, weights=mask)
        return lm_loss, lm_preds

    def get_book_logits(self, state):
        '''
        State is the state the came out from the GRU. We add some dropout, do a dense layer and then a linear layer
        with number_of_bible_books outputs.
        '''
        state = tf.nn.dropout(state, keep_prob=self.keep_prob)
        book_logits = tf.contrib.layers.fully_connected(state, num_outputs=64, activation_fn=tf.tanh)
        book_logits = tf.contrib.layers.fully_connected(book_logits, num_outputs=len(BP.book_map), activation_fn=None)
        return book_logits

    @staticmethod
    def emebd_and_run_rnn(lengths, sequence):
        '''
        Get a sequence, embed it and then run it through a GRU.
        We pass the lengths of the sequence to the dynamic rnn.
        We return the outputs for the language model and the state for the prediction of the book
        '''
        embeddings = tf.get_variable("embedding", shape=[len(BP.vocab), 32])
        emb_source = tf.nn.embedding_lookup(params=embeddings, ids=sequence, )
        cell = tf.nn.rnn_cell.GRUCell(256)
        outputs, state = tf.nn.dynamic_rnn(cell, emb_source, dtype=tf.float32, sequence_length=lengths)
        return outputs, state
