import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.wide_deep import census_dataset
from official.wide_deep import wide_deep_run_loop


class LinearClassifier(tf.estimator.Estimator):
    def __init__(self, model_dir, feature_columns, n_classes=2, config=None):

        def _model_fn(features, labels, mode, config):
            # Build the graph
            input_layer = tf.feature_column.input_layer(features, feature_columns)
            logits = tf.layers.dense(input_layer, units=1, activation=None)

            # Compute the loss
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

            # Define the train_op
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            '''手工创建EstimatorSpec'''
            #
            # # Compute the metrics
            # predicted_classes = tf.math.sigmoid(logits) > 0.5
            # acc = tf.metrics.accuracy(labels, predicted_classes, name='accuracy')
            # # auc = tf.metrics.auc(labels, logits, name='auc')
            # tf.summary.scalar('accuracy', acc)
            #
            # if mode == tf.estimator.ModeKeys.PREDICT:
            #     return tf.estimator.EstimatorSpec(mode=mode, predictions={'logits': logits})
            # elif mode == tf.estimator.ModeKeys.EVAL:
            #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'accuracy': acc})
            # else:
            #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            '''head 创建 EstimatorSpec'''
            def train_op_fn(loss):
                return optimizer.minimize(loss, global_step=tf.train.get_global_step())

            from tensorflow_estimator.python.estimator.canned import head as head_lib

            head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss()
            return head.create_estimator_spec(features, mode, logits, labels, train_op_fn=train_op_fn)

        super(LinearClassifier, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config)


def define_census_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir='/tmp/census_data',
                          model_dir='/tmp/census_model',
                          train_epochs=2,
                          epochs_between_evals=2,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0,
                          batch_size=40)


def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = model_column_fn()
  hidden_units = [100, 75, 50, 25]

  return LinearClassifier(model_dir=model_dir, feature_columns=deep_columns, n_classes=2,config=None)


def run_census(flags_obj):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """
  if flags_obj.download_if_missing:
    census_dataset.download(flags_obj.data_dir)

  train_file = os.path.join(flags_obj.data_dir, census_dataset.TRAINING_FILE)
  test_file = os.path.join(flags_obj.data_dir, census_dataset.EVAL_FILE)

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return census_dataset.input_fn(
        train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

  def eval_input_fn():
    return census_dataset.input_fn(test_file, 1, False, flags_obj.batch_size)

  tensors_to_log = {
      'average_loss': '{loss_prefix}head/truediv',
      'loss': '{loss_prefix}head/weighted_loss/Sum'
  }

  wide_deep_run_loop.run_loop(
      name="Census Income", train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      model_column_fn=census_dataset.build_model_columns,
      build_estimator_fn=build_estimator,
      flags_obj=flags_obj,
      tensors_to_log=tensors_to_log,
      early_stop=True)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_census(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_census_flags()
  absl_app.run(main)