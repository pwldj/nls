import tensorflow as tf

import modeling


def get_pretrain_loss(FLAGS, features, is_training):
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
        tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    sentence_order_labels = features["next_sentence_labels"]
    shuffle_index = features["shuffle_index"]
    seq_ids = features["seq_ids"]

    model_config = modeling.ModelConfig.from_json_file(FLAGS.model_config_file)

    model = modeling.Model(model_config, is_training, input_ids, input_mask, seq_ids, segment_ids)

    masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs = get_masked_lm_loss(model_config,
                                                                                     model.get_sequence_output(),
                                                                                     model.get_embedding_table(),
                                                                                     masked_lm_positions,
                                                                                     masked_lm_ids,
                                                                                     masked_lm_weights)

    sentence_order_loss, sentence_order_example_loss, sentence_order_log_probs = \
        get_classification_loss(model_config,
                                model.get_pooled_output(),
                                sentence_order_labels, 2)

    shuffle_loss, shuffle_example_loss, shuffle_probs = get_shuffle_loss(model_config, model.get_sequence_output(),
                                                                         shuffle_index, seq_ids)

    total_loss = masked_lm_loss + sentence_order_loss + shuffle_loss

    return total_loss, (masked_lm_example_loss, sentence_order_example_loss, shuffle_example_loss), \
           (masked_lm_log_probs, sentence_order_log_probs, shuffle_probs)


def get_shuffle_loss(model_config, seq_output, label_ids, label_weights):
    sequence_shape = modeling.get_shape_list(seq_output, expected_rank=[3])
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    seq_output = tf.reshape(seq_output, [-1, width])
    with tf.variable_scope("cls/shuffle"):
        with tf.variable_scope("transform"):
            seq_output = tf.layers.dense(
                seq_output,
                units=seq_length,
                activation=modeling.get_activation(model_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    model_config.initializer_range))
            seq_output = modeling.layer_norm(seq_output)

        output_bias = tf.get_variable(
            "output_bias",
            shape=[seq_length],
            initializer=tf.zeros_initializer())

        logits = tf.nn.bias_add(seq_output, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(tf.cast(label_weights, tf.float32), [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=seq_length, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return loss, per_example_loss, log_probs


def get_masked_lm_loss(model_config, seq_output, embedding_table, positions,
                       label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    sequence_shape = modeling.get_shape_list(seq_output, expected_rank=[3])
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(seq_output, [batch_size * seq_length, width])
    seq_output = tf.gather(flat_sequence_tensor, flat_positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                seq_output,
                units=model_config.embedding_size,
                activation=modeling.get_activation(model_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    model_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[model_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, embedding_table, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=model_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return loss, per_example_loss, log_probs


def get_classification_loss(model_config, pool_output, class_label, n_class):
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[n_class, model_config.hidden_size],
            initializer=modeling.create_initializer(
                model_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[n_class], initializer=tf.zeros_initializer())

        logits = tf.matmul(pool_output, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(class_label, [-1])
        one_hot_labels = tf.one_hot(labels, depth=n_class, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss, per_example_loss, log_probs
