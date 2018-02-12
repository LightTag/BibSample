import tensorflow as tf
import sys
from preppy import BibPreppy
import pickle
BP = pickle.load(open('./preppy.pkl','rb'))
def expand(x):
    x['length'] = tf.expand_dims(tf.convert_to_tensor(x['length']),0)
    x['book_id'] = tf.expand_dims(tf.convert_to_tensor(x['book_id']),0)
    return x
def deflate(x):
    x['length'] = tf.squeeze(x['length'])
    x['book_id'] = tf.squeeze(x['book_id'])
    return x


def make_dataset():
    dataset = tf.data.TFRecordDataset(['./val.tfrecord']).map(BibPreppy.parse)
    batch_iter = dataset.map(expand).padded_batch(64, padded_shapes={
        "book_id": 1,
        "length": 1,
        "seq": tf.TensorShape([None])
    }).map(deflate)
    next_item = batch_iter.repeat().make_one_shot_iterator().get_next()
    return next_item


class Model():
    def __init__(self, inputs):
        sequence = inputs['seq']
        lengths = inputs['length']
        book_id = inputs['book_id']
        self.lr = tf.placeholder(shape=None, dtype=tf.float32)

        emb_vec = tf.get_variable("emb", dtype=tf.float32, shape=[len(BP.vocab), 32])
        emb_source = tf.nn.embedding_lookup(emb_vec, sequence)

        cell = tf.nn.rnn_cell.GRUCell(128)
        outputs, state = tf.nn.dynamic_rnn(cell, emb_source, dtype=tf.float32, sequence_length=lengths)

        book_logits = tf.contrib.layers.fully_connected(state, num_outputs=64, activation_fn=tf.tanh)
        book_logits = tf.contrib.layers.fully_connected(book_logits, num_outputs=len(BP.book_map), activation_fn=None)

        loss = tf.losses.sparse_softmax_cross_entropy(book_id, book_logits)
        self.loss = tf.reduce_mean(loss)
        opt = tf.train.AdamOptimizer(self.lr)
        self.train = opt.minimize(self.loss)
        self.pred = tf.argmax(tf.nn.softmax(book_logits),axis=1)
        self.sequence = sequence
        self.target = book_id

if __name__=="__main__":
    ds = make_dataset()
    M = Model(ds)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    counter =0
    lr =0.01
    count =0
    epoch =0
    while True:
            count +=1
            if count %300 ==0:
                lr = lr/2
                epoch+=1
            _, loss,target,pred,inp = sess.run([M.train, M.loss,M.target, M.pred,M.sequence], feed_dict={M.lr: lr})
            print(epoch, lr,loss,pred[0],target[0])
            print(sum(pred == target) / len(target))
            if counter %100 ==0:
                print ()






