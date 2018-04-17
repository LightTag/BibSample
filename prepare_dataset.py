import tensorflow as tf

from preppy import BibPreppy


def expand(x):
    '''
    Hack. Because padded_batch doesn't play nice with scalars, so we expand the scalar to a vector of length 1
    :param x:
    :return:
    '''
    x['length'] = tf.expand_dims(tf.convert_to_tensor(x['length']), 0)
    x['book_id'] = tf.expand_dims(tf.convert_to_tensor(x['book_id']), 0)
    return x


def deflate(x):
    '''
    Undo Hack. We undo the expansion we did in expand
    '''
    x['length'] = tf.squeeze(x['length'])
    x['book_id'] = tf.squeeze(x['book_id'])
    return x


def make_dataset(path, batch_size=128):
    '''
    Makes  a Tensorflow dataset that is shuffled, batched and parsed according to BibPreppy.
    You can chain all the lines here, I split them into separate calls so I could comment easily
    :param path: The path to a tf record file
    :param path: The size of our batch
    :return: a Dataset that shuffles and is padded
    '''
    # Read a tf record file. This makes a dataset of raw TFRecords
    dataset = tf.data.TFRecordDataset([path])
    # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
    dataset = dataset.map(BibPreppy.parse, num_parallel_calls=5)
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=10000)
    # In order the pad the dataset, I had to use this hack to expand scalars to vectors.
    dataset = dataset.map(expand)
    # Batch the dataset so that we get batch_size examples in each batch.
    # Remember each item in the dataset is a dict of tensors, we need to specify padding for each tensor seperatly
    dataset = dataset.padded_batch(batch_size, padded_shapes={
        "book_id": 1,  # book_id is a scalar it doesn't need any padding, its always length one
        "length": 1,  # Likewise for the length of the sequence
        "seq": tf.TensorShape([None])  # but the seqeunce is variable length, we pass that information to TF
    })
    # Finally, we need to undo that hack from the expand function
    dataset = dataset.map(deflate)
    return dataset


def prepare_dataset_iterators(batch_size=128):
    # Make a dataset from the train data
    train_ds = make_dataset('./train.tfrecord', batch_size=batch_size)
    # make a dataset from the valdiation data
    val_ds = make_dataset('./val.tfrecord', batch_size=batch_size)
    # Define an abstract iterator
    # Make an iterator object that has the shape and type of our datasets
    iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                               train_ds.output_shapes)

    # This is an op that gets the next element from the iterator
    next_element = iterator.get_next()
    # These ops let us switch and reinitialize every time we finish an epoch
    training_init_op = iterator.make_initializer(train_ds)
    validation_init_op = iterator.make_initializer(val_ds)

    return next_element, training_init_op, validation_init_op
