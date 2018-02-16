import tensorflow as tf

from model import Model

from prepare_dataset import prepare_dataset_iterators

if __name__=="__main__":
    #Make datasets for train and validation
    next_element, training_init_op, validation_init_op = prepare_dataset_iterators()


    train_writer = tf.summary.FileWriter("./logs/train")
    val_writer = tf.summary.FileWriter("./logs/val")
    M = Model(next_element)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    counter =0
    lr =0.001
    old_val_loss,new_val_loss =100,100
    keep_prob = 1
    for epoch in range(1000):
        #Initialize the iterator to consume training data
        sess.run(training_init_op)
        if new_val_loss >old_val_loss: #We're not generalizing
            #half the learning rate
            lr = lr/2
            #And increase the amount of dropout
            keep_prob =max(keep_prob*0.9,0.3)
        while True:
            #As long as the iterator is not empty
            try:
                _, summary,gs = sess.run([M.train,M.write_op,M.gs],feed_dict={M.lr: lr, M.keep_prob:keep_prob})
                train_writer.add_summary(summary, gs)
                train_writer.flush()
            except tf.errors.OutOfRangeError:
                #If the iterator is empty stop the while loop
                break
        #Intiialize the iterator to provide validation data
        sess.run(validation_init_op)
        #We'll store the losses from each batch to get an average
        losses = []
        while True:
            # As long as the iterator is not empty
            try:
                loss,summary,gs,_ = sess.run([M.total_loss,M.write_op,M.gs,M.increment_gs],feed_dict={M.lr: lr,M.keep_prob:1})
                losses.append(loss)
                val_writer.add_summary(summary, gs)
                val_writer.flush()
            except tf.errors.OutOfRangeError:
                #Update the average loss for the epoch
                old_val_loss = new_val_loss
                new_val_loss =sum(losses)/len(losses)
                break







