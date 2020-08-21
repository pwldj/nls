import os

import tensorflow as tf
import classifier_utils
import modeling
import tensorflow.compat.v1 as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu

import tokenization

flags = tf.flags

flags.DEFINE_string("model_config_file", None,
                    "")
flags.DEFINE_string("model_dir", None,
                    "")
flags.DEFINE_string("init_checkpoint", None,
                    "")
flags.DEFINE_integer("max_seq_length", 512,
                     "")

# Optimization
flags.DEFINE_integer("num_core_per_host", 1,
                     "")
flags.DEFINE_float("learning_rate", 5e-5,
                   "initial learning rate")
flags.DEFINE_float("min_lr_ratio", 0.0,
                   "min lr ratio for cos decay.")
flags.DEFINE_float("clip", 1.0,
                   "Gradient clipping")
flags.DEFINE_float("weight_decay", 0.00,
                   "Weight decay rate")
flags.DEFINE_float("adam_epsilon", 1e-6,
                   "Adam epsilon")
flags.DEFINE_string("decay_method", "poly",
                    "poly or cos")
flags.DEFINE_string("data_dir", None,
                    "")
flags.DEFINE_string("cache_dir", None,
                    "")

# Training
flags.DEFINE_bool("do_train", True,
                  "whether to do training")
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
flags.DEFINE_bool("do_eval", True,
                  "whether to do eval")
flags.DEFINE_bool("do_predict", True,
                  "whether to do predict")
flags.DEFINE_integer("eval_batch_size", 32,
                     "batch size for eval")
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


def serving_input_receiver_fn():
    """Creates an input function for serving."""
    seq_len = FLAGS.max_seq_length
    serialized_example = tf.placeholder(
        dtype=tf.string, shape=[None], name="serialized_example")
    features = {
        "input_ids": tf.FixedLenFeature([seq_len], dtype=tf.int64),
        "input_mask": tf.FixedLenFeature([seq_len], dtype=tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_len], dtype=tf.int64),
    }
    feature_map = tf.parse_example(serialized_example, features=features)
    feature_map["is_real_example"] = tf.constant(1, dtype=tf.int32)
    feature_map["label_ids"] = tf.constant(0, dtype=tf.int32)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in feature_map.keys():
        t = feature_map[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        feature_map[name] = t

    return tf.estimator.export.ServingInputReceiver(
        features=feature_map, receiver_tensors=serialized_example)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "cola": classifier_utils.ColaProcessor,
        "mnli": classifier_utils.MnliProcessor,
        "mismnli": classifier_utils.MisMnliProcessor,
        "mrpc": classifier_utils.MrpcProcessor,
        "rte": classifier_utils.RteProcessor,
        "sst-2": classifier_utils.Sst2Processor,
        "sts-b": classifier_utils.StsbProcessor,
        "qqp": classifier_utils.QqpProcessor,
        "qnli": classifier_utils.QnliProcessor,
        "wnli": classifier_utils.WnliProcessor,
    }

    if not (FLAGS.do_train or FLAGS.do_eval or FLAGS.do_predict or
            FLAGS.export_dir):
        raise ValueError(
            "At least one of `do_train`, `do_eval`, `do_predict' or `export_dir` "
            "must be True.")

    if not FLAGS.albert_config_file and not FLAGS.albert_hub_module_handle:
        raise ValueError("At least one of `--albert_config_file` and "
                         "`--albert_hub_module_handle` must be set")

    if FLAGS.albert_config_file:
        albert_config = modeling.AlbertConfig.from_json_file(
            FLAGS.albert_config_file)
        if FLAGS.max_seq_length > albert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the ALBERT model "
                "was only trained up to sequence length %d" %
                (FLAGS.max_seq_length, albert_config.max_position_embeddings))
    else:
        albert_config = None  # Get the config from TF-Hub.

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](
        use_spm=True if FLAGS.spm_model_file else False,
        do_lower_case=FLAGS.do_lower_case)

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer.from_scratch(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
        spm_model_file=FLAGS.spm_model_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
    if FLAGS.do_train:
        iterations_per_loop = int(min(FLAGS.iterations_per_loop,
                                      FLAGS.save_checkpoints_steps))
    else:
        iterations_per_loop = FLAGS.iterations_per_loop
    run_config = contrib_tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=int(FLAGS.save_checkpoints_steps),
        keep_checkpoint_max=0,
        tpu_config=contrib_tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
    model_fn = classifier_utils.model_fn_builder(
        albert_config=albert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.train_step,
        num_warmup_steps=FLAGS.warmup_step,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        task_name=task_name,
        hub_module=FLAGS.albert_hub_module_handle,
        optimizer=FLAGS.optimizer)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = contrib_tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        export_to_tpu=False)  # http://yaqs/4707241341091840

    if FLAGS.do_train:
        cached_dir = FLAGS.cached_dir
        if not cached_dir:
            cached_dir = FLAGS.output_dir
        train_file = os.path.join(cached_dir, task_name + "_train.tf_record")
        if not tf.gfile.Exists(train_file):
            classifier_utils.file_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_length, tokenizer,
                train_file, task_name)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", FLAGS.train_step)
        train_input_fn = classifier_utils.file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
            task_name=task_name,
            use_tpu=FLAGS.use_tpu,
            bsz=FLAGS.train_batch_size)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_step)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(classifier_utils.PaddingInputExample())

        cached_dir = FLAGS.cached_dir
        if not cached_dir:
            cached_dir = FLAGS.output_dir
        eval_file = os.path.join(cached_dir, task_name + "_eval.tf_record")
        if not tf.gfile.Exists(eval_file):
            classifier_utils.file_based_convert_examples_to_features(
                eval_examples, label_list, FLAGS.max_seq_length, tokenizer,
                eval_file, task_name)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = classifier_utils.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            task_name=task_name,
            use_tpu=FLAGS.use_tpu,
            bsz=FLAGS.eval_batch_size)

        best_trial_info_file = os.path.join(FLAGS.output_dir, "best_trial.txt")

        def _best_trial_info():
            """Returns information about which checkpoints have been evaled so far."""
            if tf.gfile.Exists(best_trial_info_file):
                with tf.gfile.GFile(best_trial_info_file, "r") as best_info:
                    global_step, best_metric_global_step, metric_value = (
                        best_info.read().split(":"))
                    global_step = int(global_step)
                    best_metric_global_step = int(best_metric_global_step)
                    metric_value = float(metric_value)
            else:
                metric_value = -1
                best_metric_global_step = -1
                global_step = -1
            tf.logging.info(
                "Best trial info: Step: %s, Best Value Step: %s, "
                "Best Value: %s", global_step, best_metric_global_step, metric_value)
            return global_step, best_metric_global_step, metric_value

        def _remove_checkpoint(checkpoint_path):
            for ext in ["meta", "data-00000-of-00001", "index"]:
                src_ckpt = checkpoint_path + ".{}".format(ext)
                tf.logging.info("removing {}".format(src_ckpt))
                tf.gfile.Remove(src_ckpt)

        def _find_valid_cands(curr_step):
            filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
            candidates = []
            for filename in filenames:
                if filename.endswith(".index"):
                    ckpt_name = filename[:-6]
                    idx = ckpt_name.split("-")[-1]
                    if int(idx) > curr_step:
                        candidates.append(filename)
            return candidates

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")

        if task_name == "sts-b":
            key_name = "pearson"
        elif task_name == "cola":
            key_name = "matthew_corr"
        else:
            key_name = "eval_accuracy"

        global_step, best_perf_global_step, best_perf = _best_trial_info()
        writer = tf.gfile.GFile(output_eval_file, "w")
        while global_step < FLAGS.train_step:
            steps_and_files = {}
            filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
            for filename in filenames:
                if filename.endswith(".index"):
                    ckpt_name = filename[:-6]
                    cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
                    if cur_filename.split("-")[-1] == "best":
                        continue
                    gstep = int(cur_filename.split("-")[-1])
                    if gstep not in steps_and_files:
                        tf.logging.info("Add {} to eval list.".format(cur_filename))
                        steps_and_files[gstep] = cur_filename
            tf.logging.info("found {} files.".format(len(steps_and_files)))
            if not steps_and_files:
                tf.logging.info("found 0 file, global step: {}. Sleeping."
                                .format(global_step))
                time.sleep(60)
            else:
                for checkpoint in sorted(steps_and_files.items()):
                    step, checkpoint_path = checkpoint
                    if global_step >= step:
                        if (best_perf_global_step != step and
                                len(_find_valid_cands(step)) > 1):
                            _remove_checkpoint(checkpoint_path)
                        continue
                    result = estimator.evaluate(
                        input_fn=eval_input_fn,
                        steps=eval_steps,
                        checkpoint_path=checkpoint_path)
                    global_step = result["global_step"]
                    tf.logging.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        tf.logging.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                    writer.write("best = {}\n".format(best_perf))
                    if result[key_name] > best_perf:
                        best_perf = result[key_name]
                        best_perf_global_step = global_step
                    elif len(_find_valid_cands(global_step)) > 1:
                        _remove_checkpoint(checkpoint_path)
                    writer.write("=" * 50 + "\n")
                    writer.flush()
                    with tf.gfile.GFile(best_trial_info_file, "w") as best_info:
                        best_info.write("{}:{}:{}".format(
                            global_step, best_perf_global_step, best_perf))
        writer.close()

        for ext in ["meta", "data-00000-of-00001", "index"]:
            src_ckpt = "model.ckpt-{}.{}".format(best_perf_global_step, ext)
            tgt_ckpt = "model.ckpt-best.{}".format(ext)
            tf.logging.info("saving {} to {}".format(src_ckpt, tgt_ckpt))
            tf.io.gfile.rename(
                os.path.join(FLAGS.output_dir, src_ckpt),
                os.path.join(FLAGS.output_dir, tgt_ckpt),
                overwrite=True)

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(classifier_utils.PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        classifier_utils.file_based_convert_examples_to_features(
            predict_examples, label_list,
            FLAGS.max_seq_length, tokenizer,
            predict_file, task_name)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = classifier_utils.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
            task_name=task_name,
            use_tpu=FLAGS.use_tpu,
            bsz=FLAGS.predict_batch_size)

        checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt-best")
        result = estimator.predict(
            input_fn=predict_input_fn,
            checkpoint_path=checkpoint_path)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        output_submit_file = os.path.join(FLAGS.output_dir, "submit_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as pred_writer, \
                tf.gfile.GFile(output_submit_file, "w") as sub_writer:
            sub_writer.write("index" + "\t" + "prediction\n")
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, (example, prediction)) in \
                    enumerate(zip(predict_examples, result)):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                pred_writer.write(output_line)

                if task_name != "sts-b":
                    actual_label = label_list[int(prediction["predictions"])]
                else:
                    actual_label = str(prediction["predictions"])
                sub_writer.write(example.guid + "\t" + actual_label + "\n")
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

    if FLAGS.export_dir:
        tf.gfile.MakeDirs(FLAGS.export_dir)
        checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt-best")
        tf.logging.info("Starting to export model.")
        subfolder = estimator.export_saved_model(
            export_dir_base=FLAGS.export_dir,
            serving_input_receiver_fn=serving_input_receiver_fn,
            checkpoint_path=checkpoint_path)
        tf.logging.info("Model exported to %s.", subfolder)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("spm_model_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
