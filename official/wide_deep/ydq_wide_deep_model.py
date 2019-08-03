import tensorflow as tf


class LinearClassifier(tf.estimator.Estimator):
    def __init__(self, model_dir, feature_columns, n_classes=2, config=None):

        def _model_fn(features, labels, mode, config):
            # Build the graph
            input_layer = tf.feature_column.input_layer(features, feature_columns)
            logits = tf.layers.dense(input_layer, units=n_classes, activation=tf.nn.sigmoid)

            # Compute the predicted class
            predicted_classes = tf.math.argmax(logits, axis=1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_id': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits
                }
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

            # Compute the loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

            # Compute the metrics
            acc = tf.metrics.accuracy(labels, predicted_classes, name='accuracy')
            # auc = tf.metrics.auc(labels, logits, name='auc')
            tf.summary.scalar('accuracy', acc[1])
            # tf.summary.scalar('auc', auc)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={'accuracy': acc,
                                                                              # 'auc': auc
                                                                              })

            # Define the optimizer
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        super(LinearClassifier, self).__init__(model_fn=_model_fn, model_dir=model_dir, config=config)