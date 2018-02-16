import tensorflow as tf

from model import Model,BP

from prepare_dataset import prepare_dataset_iterators

if __name__=="__main__":
    #Make datasets for train and validation
    with tf.Graph().as_default():
        gs = tf.train.get_or_create_global_step()
        next_element, training_init_op, validation_init_op = prepare_dataset_iterators(batch_size=64)

        train_writer = tf.summary.FileWriter("./logs/train")
        val_writer = tf.summary.FileWriter("./logs/val")
        M = Model(next_element,gs=gs)
        init =tf.global_variables_initializer()
        with tf.train.MonitoredTrainingSession(checkpoint_dir="./chkpoint",
                                               save_summaries_steps=None,
                                               save_summaries_secs=None, ) as sess:


            sess.run(init)
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
                        if gs %20 ==0:
                            inp,out,lengths = sess.run([M.sequence,M.lm_preds,M.lengths],feed_dict={M.lr: lr, M.keep_prob:keep_prob})
                            print("{gs}************************************************".format(gs=gs))
                            print(BP.ids_to_string(inp[0],lengths[0]))
                            print("<start>"+BP.ids_to_string(out[0], lengths[0]))

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







