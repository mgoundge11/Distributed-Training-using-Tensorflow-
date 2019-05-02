import argparse
import sys
import math

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None
IMAGE_PIXELS=28

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      hid_w = tf.Variable(
          tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                              stddev=1.0 / IMAGE_PIXELS), name="hid_w")
      hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

      # Variables of the softmax layer
      sm_w = tf.Variable(
          tf.truncated_normal([FLAGS.hidden_units, 10],
                              stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
          name="sm_w")
      sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
      y_ = tf.placeholder(tf.float32, [None, 10])

      hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      hid = tf.nn.relu(hid_lin)

      y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
      loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
      #global_step = tf.contrib.framework.get_or_create_global_step()
      global_step = tf.train.get_or_create_global_step()
      train_op = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss, global_step=global_step)

      # train_op = tf.train.AdagradOptimizer(0.01).minimize(
      #     loss, global_step=global_step)

    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()
    init_op = tf.initialize_all_variables()
    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=100)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir=FLAGS.model_dir,
                                           hooks=hooks) as mon_sess:
      step = 0
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See <a href="./../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.

        batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
        train_feed = {x: batch_xs, y_: batch_ys}

        # _, step = mon_sess.run([train_op, global_step], feed_dict=train_feed)
        mon_sess.run([train_op, global_step], feed_dict=train_feed)
        #mon_sess.run(train_op, feed_dict=train_feed)
        # if step % 100 == 0:
        #     print("Done step %d" % step)
        #mon_sess.run(train_op)
        print("Training")
    saver.save(mon_sess, FLAGS.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=100,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./MNIST_data",
        help="Index of task within the job"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Index of task within the job"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="dir to save the trained model"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()

