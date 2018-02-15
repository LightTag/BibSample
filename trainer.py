import tensorflow as tf
import sys

from tensorflow.contrib.layers import xavier_initializer
from tensor2tensor.layers.common_layers import embedding
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


def make_dataset(path):
    dataset = tf.data.TFRecordDataset([path]).map(BibPreppy.parse,num_parallel_calls=5).shuffle(buffer_size=10000)
    batch_iter = dataset.map(expand).padded_batch(32, padded_shapes={
        "book_id": 1,
        "length": 1,
        "seq": tf.TensorShape([None])
    }).map(deflate)
    next_item = batch_iter
    return next_item


class Model():
    def __init__(self, inputs):
        sequence = inputs['seq']
        lengths = inputs['length']
        book_id = inputs['book_id']
        self.lr = tf.placeholder(shape=None, dtype=tf.float32)
        self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32)
        with tf.variable_scope("main",initializer=xavier_initializer()):
            emb_source =embedding(sequence,vocab_size=len(BP.vocab),dense_size=32)

            cell = tf.nn.rnn_cell.GRUCell(256)
            outputs, state = tf.nn.dynamic_rnn(cell, emb_source, dtype=tf.float32, sequence_length=lengths)
            state = tf.nn.dropout(state,keep_prob=self.keep_prob)
            book_logits = tf.contrib.layers.fully_connected(state, num_outputs=64, activation_fn=tf.tanh)
            book_logits = tf.contrib.layers.fully_connected(book_logits, num_outputs=len(BP.book_map), activation_fn=None)

            loss = tf.losses.sparse_softmax_cross_entropy(book_id, book_logits)

            lm_target =sequence[:,1:]
            outputs = outputs[:,:-1,:]
            lm_logits =  tf.contrib.layers.fully_connected(outputs, num_outputs=len(BP.vocab), activation_fn=None)
            mask = tf.sequence_mask(lengths-1,tf.reduce_max(lengths)-1)
            lm_loss = tf.losses.sparse_softmax_cross_entropy(lm_target, lm_logits,weights=mask)



            self.total_loss = tf.reduce_mean(loss) + tf.reduce_mean(lm_loss)
            opt = tf.train.AdamOptimizer(self.lr)
            self.train = opt.minimize(self.total_loss)
            self.pred = tf.argmax(tf.nn.softmax(book_logits),axis=1)
            self.sequence = sequence
            self.target = book_id
            self.accuracy =tf.contrib.metrics.accuracy(predictions=self.pred,labels=self.target)

            tf.summary.scalar("batch_accuracy",self.accuracy)
            tf.summary.scalar("total loss", self.total_loss)
            tf.summary.scalar("lm loss", lm_loss)
            tf.summary.scalar("book loss", loss)
            tf.summary.scalar("lr", self.lr)
            self.write_op = tf.summary.merge_all()

class TransModel():
    pass
if __name__=="__main__":
    train_ds = make_dataset('./train.tfrecord')
    val_ds = make_dataset('./val.tfrecord')
    iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                               train_ds.output_shapes)
    train_writer = tf.summary.FileWriter("./logs/train1")
    val_writer  = tf.summary.FileWriter("./logs/val1")

    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(train_ds)
    validation_init_op = iterator.make_initializer(val_ds)

    M = Model(next_element)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    counter =0
    lr =0.001
    count =0
    old_val_loss,new_val_loss =100,100
    for epoch in range(1000):
        sess.run(training_init_op)
        if new_val_loss >old_val_loss:
            lr = lr/2
        while True:
            count+=1
            try:
                _, loss,target,pred,inp,summary = sess.run([M.train, M.total_loss,M.target, M.pred,M.sequence,M.write_op],
                                                           feed_dict={M.lr: lr, M.keep_prob:0.9})
                train_writer.add_summary(summary, count)
                train_writer.flush()
            except tf.errors.OutOfRangeError:
                break
        sess.run(validation_init_op)
        losses = []
        while True:
            count += 1

            try:
                loss,target,pred,inp,summary = sess.run([M.total_loss,M.target, M.pred,M.sequence,M.write_op],
                                                        feed_dict={M.lr: lr,M.keep_prob:1})
                losses.append(loss)
                val_writer.add_summary(summary, count)
                val_writer.flush()
            except tf.errors.OutOfRangeError:
                old_val_loss = new_val_loss
                new_val_loss =sum(losses)/len(losses)
                break







