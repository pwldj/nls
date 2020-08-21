import json
import os

import numpy as np
import tensorflow as tf
import function_builder
import model_util

flags = tf.flags

flags.DEFINE_string("model_config_file", None,
                    "")
flags.DEFINE_string("model_dir", None,
                    "")
flags.DEFINE_string("init_checkpoint", None,
                    "")
flags.DEFINE_integer("max_seq_length", 512,
                     "")
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "")

# Optimization
flags.DEFINE_integer("num_core_per_host", 1,
                     "")
flags.DEFINE_float("learning_rate", 0.0002,
                   "initial learning rate")
flags.DEFINE_float("min_lr_ratio", 0.0,
                   "min lr ratio for cos decay.")
flags.DEFINE_float("clip", 1.0,
                   "Gradient clipping")
flags.DEFINE_float("weight_decay", 0.01,
                   "Weight decay rate")
flags.DEFINE_float("adam_epsilon", 1e-6,
                   "Adam epsilon")
flags.DEFINE_string("decay_method", "poly",
                    "poly or cos")

# Training
flags.DEFINE_bool("do_train", True,
                  "whether to do training")
flags.DEFINE_string("train_dir", None,
                    "")
flags.DEFINE_integer("train_batch_size", 4,
                     "batch size for training")
flags.DEFINE_integer("train_steps", 90000,
                     "Number of training steps")
flags.DEFINE_integer("warmup_steps", 1000,
                     "number of warmup steps")
flags.DEFINE_integer("save_steps", 3000,
                     "Save the model for every save_steps. "
                     "If None, not to save any model.")
flags.DEFINE_integer("max_save", 5,
                     "Max number of checkpoints to save. "
                     "Use 0 to save all.")
flags.DEFINE_integer("shuffle_buffer", 4096,
                     "Buffer size used for shuffle.")

# Eval / Prediction
flags.DEFINE_bool("do_eval", False,
                  "whether to do eval")
flags.DEFINE_bool("do_predict", False,
                  "whether to do predict")
flags.DEFINE_integer("eval_batch_size", 32,
                     "batch size for eval")
flags.DEFINE_integer("eval_steps", 100,
                     "do eval steps")
flags.DEFINE_string("eval_file", None,
                    "")
flags.DEFINE_string("predict_file", None,
                    "")
flags.DEFINE_string("predict_dir", None,
                    "")
flags.DEFINE_integer("predict_batch_size", 1,
                     "")

# TPU
flags.DEFINE_bool("use_tpu", False,
                  "")
flags.DEFINE_string("master", None,
                    "")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_integer("iterations", default=1000,
                     help="number of iterations per TPU training loop.")

FLAGS = flags.FLAGS


def model_fn_builder():
    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        total_loss, per_example_loss, logits = function_builder.get_pretrain_loss(
            FLAGS, features, is_training)

        # Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # load pretrained models
        scaffold_fn = model_util.init_from_checkpoint(FLAGS)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op, learning_rate, _ = model_util.get_train_op(
                FLAGS, total_loss)
            if FLAGS.use_tpu:
                raise Exception("not support")
            else:
                train_spec = tf.estimator.EstimatorSpec(
                    mode=mode, loss=total_loss, train_op=train_op)
            return train_spec

        if mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, features, logits):
                logits_mask = tf.argmax(
                    logits[0], axis=-1, output_type=tf.int32)
                logits_order = tf.argmax(
                    logits[1], axis=-1, output_type=tf.int32)
                logits_shuffle = tf.argmax(
                    logits[2], axis=-1, output_type=tf.int32)
                acc_mask = tf.metrics.accuracy(
                    labels=features["masked_lm_ids"], predictions=logits_mask)
                acc_order = tf.metrics.accuracy(
                    labels=features["next_sentence_labels"], predictions=logits_order)
                acc_shuffle = tf.metrics.accuracy(
                    labels=features["shuffle_index"], predictions=logits_shuffle)
                loss = tf.metrics.mean(values=per_example_loss)

                return {"mask_accuracy": acc_mask, "order_accuracy": acc_order, "shuffle_accuracy": acc_shuffle,
                        "eval_loss": loss}

            if FLAGS.use_tpu:
                raise Exception("not support")
            else:
                eval_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=metric_fn(per_example_loss, features, logits))
            return eval_spec

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"masked_lm_ids": features["masked_lm_ids"],
                           "masked_lm_logits": tf.reshape(logits[0], [1, -1]),
                           "next_sentence_labels": features["next_sentence_labels"],
                           "next_sentence_logits": logits[1],
                           "shuffle_labels": features["shuffle_index"],
                           "shuffle_logits": tf.reshape(logits[2], [1, -1])}
            if FLAGS.use_tpu:
                raise Exception("not support")
            else:
                predict_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions
                )
            return predict_spec

    return model_fn


def input_fn_builder(input_file, max_seq_length, max_predictions_per_seq, is_training, drop_remainder=True):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "next_sentence_labels": tf.FixedLenFeature([1], tf.int64),
        "masked_lm_positions": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights": tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "shuffle_index": tf.FixedLenFeature([max_seq_length], tf.int64),
        "seq_ids": tf.FixedLenFeature([max_seq_length], tf.int64)}

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t
        return example

    def input_fn(params, input_context=None):
        if FLAGS.use_tpu:
            batch_size = params["batch_size"]
        elif is_training:
            batch_size = FLAGS.train_batch_size
        elif FLAGS.do_eval:
            batch_size = FLAGS.eval_batch_size
        else:
            batch_size = FLAGS.predict_batch_size

        d = tf.data.TFRecordDataset(input_file)

        if input_context is not None:
            tf.logging.info("Input pipeline id %d out of %d",
                            input_context.input_pipeline_id, input_context.num_replicas_in_sync)
            d = d.shard(input_context.num_input_pipelines,
                        input_context.input_pipeline_id)

        if is_training:
            d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
            d = d.repeat()
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    run_config = model_util.configure_tpu(FLAGS)
    model_fn = model_fn_builder()

    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            eval_batch_size=FLAGS.eval_batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    if FLAGS.do_train:
        train_files = [os.path.join(FLAGS.train_dir, x) for x in os.listdir(FLAGS.train_dir)]
        train_input_fn = input_fn_builder(
            input_file=train_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    if FLAGS.do_eval:
        eval_input_fn = input_fn_builder(
            input_file=FLAGS.eval_file,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False,
            drop_remainder=True
        )
        # Filter out all checkpoints in the directory
        steps_and_files = []
        filenames = tf.gfile.ListDirectory(FLAGS.model_dir)
        for filename in filenames:
            if filename.endswith(".index"):
                ckpt_name = filename[:-6]
                cur_filename = os.path.join(FLAGS.model_dir, ckpt_name)
                global_step = int(cur_filename.split("-")[-1])
                tf.logging.info("Add {} to eval list.".format(cur_filename))
                steps_and_files.append([global_step, cur_filename])
        steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

        # Decide whether to evaluate all ckpts
        if not FLAGS.eval_all_ckpt:
            steps_and_files = steps_and_files[-1:]

        eval_results = []
        for global_step, filename in steps_and_files:
            ret = estimator.evaluate(
                input_fn=eval_input_fn,
                checkpoint_path=filename)

            ret["step"] = global_step
            ret["path"] = filename

            eval_results.append(ret)

            tf.logging.info("=" * 80)
            log_str = "Eval result | "
            for key, val in sorted(ret.items(), key=lambda x: x[0]):
                log_str += "{} {} | ".format(key, val)
            tf.logging.info(log_str)

    if FLAGS.do_predict:
        predict_input_fn = input_fn_builder(
            input_file=FLAGS.predict_dir,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False,
            drop_remainder=True
        )
        result = []
        for r in estimator.predict(predict_input_fn,
                                   yield_single_examples=True,
                                   checkpoint_path=FLAGS.init_checkpoint):
            result.append(r)
            print(r)

        predict_json_path = os.path.join(FLAGS.predict_dir, "pretrain_logits.json")
        with tf.gfile.Open(predict_json_path, "w") as fp:
            json.dump(result, fp, indent=4)


if __name__ == "__main__":
    tf.app.run()
