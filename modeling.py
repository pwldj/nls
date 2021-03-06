import copy
import json
import math

import six
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers


class ModelConfig(object):
    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 hidden_size=4096,
                 num_hidden_layers=12,
                 num_hidden_groups=1,
                 num_attention_heads=64,
                 intermediate_size=16384,
                 hidden_act="gelu",
                 hidden_dropout_prob=0,
                 attention_probs_dropout_prob=0,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02
                 ):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `ModelConfig` from a Python dictionary of parameters."""
        config = ModelConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `ModelConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=4, sort_keys=True) + "\n"


class Model:
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 seq_type_ids=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=False,
                 scope=None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(
                shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(
                shape=[batch_size, seq_length], dtype=tf.int32)

        with tf.variable_scope(scope, default_name='bert'):
            with tf.variable_scope('embedding'):
                (self.word_embedding_output,
                 self.output_embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.embedding_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name='word_embeddings',
                    use_one_hot_embeddings=use_one_hot_embeddings)

                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.word_embedding_output,
                    use_token_type=True,
                    seq_type_ids=seq_type_ids,
                    token_type_ids=token_type_ids,
                    seq_type_embedding_name="seq_type_embeddings",
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name='token_type_embeddings',
                    use_position_embeddings=True,
                    position_embedding_name='position_embeddings',
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob,
                    use_one_hot_embeddings=use_one_hot_embeddings)

            with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
                self.all_encoder_layers = transformer_model(
                    input_tensor=self.embedding_output,
                    attention_mask=input_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_hidden_groups=config.num_hidden_groups,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=get_activation(config.hidden_act),
                    hidden_dropout_prob=config.hidden_dropout_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True)

            self.sequence_output = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(
                    self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_word_embedding_output(self):
        return self.word_embedding_output

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.output_embedding_table


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])

    embedding_table = tf.get_variable(
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))

    if use_one_hot_embeddings:
        flat_input_ids = tf.reshape(input_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        output = tf.nn.embedding_lookup(embedding_table, input_ids)

    input_shape = get_shape_list(input_ids)

    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return output, embedding_table


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            seq_type_ids=None,
                            token_type_ids=None,
                            seq_type_embedding_name="seq_type_embeddings",
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1,
                            use_one_hot_embeddings=True):
    input_shape = get_shape_list(input_tensor, expected_rank=[3])
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.get_variable(
            name=token_type_embedding_name,
            shape=[token_type_vocab_size, width],
            initializer=create_initializer(initializer_range))
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary, unless converting to tflite model.
        if use_one_hot_embeddings:
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])
            one_hot_ids = tf.one_hot(
                flat_token_type_ids, depth=token_type_vocab_size)
            token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
            token_type_embeddings = tf.reshape(token_type_embeddings,
                                               [batch_size, seq_length, width])
        else:
            token_type_embeddings = tf.nn.embedding_lookup(token_type_table,
                                                           token_type_ids)
        output += token_type_embeddings

    if seq_type_ids is not None:
        seq_type_table = tf.get_variable(
            name=seq_type_embedding_name,
            shape=[seq_length, width],
            initializer=create_initializer(initializer_range))
        seq_type_embeddings = tf.nn.embedding_lookup(seq_type_table,
                                                     token_type_ids)
        output += seq_type_embeddings

    if use_position_embeddings:
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name,
                shape=[max_position_embeddings, width],
                initializer=create_initializer(initializer_range))
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings,
                                             position_broadcast_shape)
            output += position_embeddings

    output = layer_norm_and_dropout(output, dropout_prob)
    return output


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_hidden_groups=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn="gelu",
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      in_group_reuse=True):
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size {} is not a multiple of the number of attention "
            "heads {}".format(hidden_size, num_attention_heads))

    if num_hidden_layers % num_hidden_groups != 0:
        raise ValueError(
            "number hidden layers {} is not a multiple of the number of num "
            "hidden groups {}".format(num_hidden_layers, num_attention_heads))

    attention_head_size = hidden_size // num_attention_heads
    input_shape = get_shape_list(input_tensor, expected_rank=[3])
    input_width = input_shape[2]

    all_layer_outputs = []
    if input_width != hidden_size:
        prev_output = abc_cd_abd(
            input_tensor, hidden_size, create_initializer(initializer_range),
            None, name="embedding_hidden_mapping_in")
    else:
        prev_output = input_tensor

    num_layers_pre_group = int(num_hidden_layers / num_hidden_groups)
    for group_idx in range(num_hidden_groups):
        with tf.name_scope("group_%d" % group_idx):
            for inner_group_idx in range(num_layers_pre_group):
                layer_idx = group_idx * num_layers_pre_group + inner_group_idx
                var_idx = group_idx if in_group_reuse else inner_group_idx
                with tf.name_scope("layer_%d" % layer_idx):
                    with tf.variable_scope("inner_layer_%d" % var_idx):
                        layer_output = prev_output
                        layer_output = attention_ffn_block(
                            layer_input=layer_output,
                            hidden_size=hidden_size,
                            attention_mask=attention_mask,
                            num_attention_heads=num_attention_heads,
                            attention_head_size=attention_head_size,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            intermediate_size=intermediate_size,
                            intermediate_act_fn=intermediate_act_fn,
                            initializer_range=initializer_range,
                            hidden_dropout_prob=hidden_dropout_prob)
                        prev_output = layer_output
                        all_layer_outputs.append(layer_output)
    if do_return_all_layers:
        return all_layer_outputs
    else:
        return all_layer_outputs[-1]


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None,
                    use_einsum=True):
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])
    size_per_head = int(from_shape[2] / num_attention_heads)

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if batch_size is None or from_seq_length is None or to_seq_length is None:
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # `query_layer` = [B, F, N, H]
    q = abc_ced_abde(from_tensor, num_attention_heads, size_per_head,
                     create_initializer(initializer_range), query_act, "query")

    # `key_layer` = [B, T, N, H]
    k = abc_ced_abde(to_tensor, num_attention_heads, size_per_head,
                     create_initializer(initializer_range), key_act, "key")
    # `value_layer` = [B, T, N, H]
    v = abc_ced_abde(to_tensor, num_attention_heads, size_per_head,
                     create_initializer(initializer_range), value_act, "value")
    q = tf.transpose(q, [0, 2, 1, 3])
    k = tf.transpose(k, [0, 2, 1, 3])
    v = tf.transpose(v, [0, 2, 1, 3])
    if attention_mask is not None:
        attention_mask = tf.reshape(
            attention_mask, [batch_size, 1, to_seq_length, 1])
        # 'new_embeddings = [B, N, F, H]'
    new_embeddings = dot_product_attention(q, k, v, attention_mask,
                                           attention_probs_dropout_prob)

    return tf.transpose(new_embeddings, [0, 2, 1, 3])


def attention_ffn_block(layer_input,
                        hidden_size=768,
                        attention_mask=None,
                        num_attention_heads=1,
                        attention_head_size=64,
                        attention_probs_dropout_prob=0.0,
                        intermediate_size=3072,
                        intermediate_act_fn=None,
                        initializer_range=0.02,
                        hidden_dropout_prob=0.0):
    with tf.variable_scope("attention_1"):
        with tf.variable_scope("self"):
            attention_output = attention_layer(
                from_tensor=layer_input,
                to_tensor=layer_input,
                attention_mask=attention_mask,
                num_attention_heads=num_attention_heads,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
            attention_output = abcd_cde_abe(
                attention_output,
                hidden_size,
                attention_head_size,
                create_initializer(initializer_range),
                None,
                name="dense")
            attention_output = dropout(attention_output, hidden_dropout_prob)
    attention_output = layer_norm(attention_output + layer_input)
    with tf.variable_scope("ffn_1"):
        with tf.variable_scope("intermediate"):
            intermediate_output = abc_cd_abd(
                attention_output,
                intermediate_size,
                create_initializer(initializer_range),
                intermediate_act_fn,
                name="dense")
            with tf.variable_scope("output"):
                ffn_output = abc_cd_abd(
                    intermediate_output,
                    hidden_size,
                    create_initializer(initializer_range),
                    None,
                    name="dense")
            ffn_output = dropout(ffn_output, hidden_dropout_prob)
    ffn_output = layer_norm(ffn_output + attention_output)
    return ffn_output


def dot_product_attention(q, k, v, bias, dropout_rate=0.0):
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    logits = tf.multiply(logits, 1.0 / math.sqrt(float(get_shape_list(q)[-1])))
    if bias is not None:
        # `attention_mask` = [B, T]
        from_shape = get_shape_list(q)
        broadcast_ones = tf.ones(
            [from_shape[0], 1, from_shape[2], 1], tf.float32)

        bias = tf.matmul(broadcast_ones,
                         tf.cast(bias, tf.float32), transpose_b=True)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - bias) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        logits += adder
    else:
        adder = 0.0

    attention_probs = tf.nn.softmax(logits, name="attention_probs")
    attention_probs = dropout(attention_probs, dropout_rate)
    return tf.matmul(attention_probs, v)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor, name)
    output_tensor = dropout(output_tensor, dropout_prob)
    return output_tensor


def abc_cd_abd(input_tensor,
               output_size,
               initializer,
               activation,
               name=None):
    input_shape = get_shape_list(input_tensor)
    hidden_size = input_shape[2]
    with tf.variable_scope(name):
        w = tf.get_variable(
            name="kernel",
            shape=[hidden_size, output_size],
            initializer=initializer)
        b = tf.get_variable(
            name="bias", shape=[output_size], initializer=tf.zeros_initializer)
        ret = tf.einsum("BFH,HO->BFO", input_tensor, w)
        ret += b
    if activation is not None:
        return activation(ret)
    else:
        return ret


def abcd_cde_abe(input_tensor,
                 hidden_size,
                 head_size,
                 initializer,
                 activation,
                 name=None):
    """A dense layer with 3D kernel for projection.

  Args:
    input_tensor: float Tensor of shape [batch,from_seq_length,
      num_attention_heads, size_per_head].
    hidden_size: The size of hidden layer.
    head_size: The size of head.
    initializer: Kernel initializer.
    activation: Actication function.
    use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """
    input_shape = get_shape_list(input_tensor)
    num_attention_heads = input_shape[2]
    with tf.variable_scope(name):
        w = tf.get_variable(
            name="kernel",
            shape=[num_attention_heads * head_size, hidden_size],
            initializer=initializer)
        w = tf.reshape(w, [num_attention_heads, head_size, hidden_size])
        b = tf.get_variable(
            name="bias", shape=[hidden_size], initializer=tf.zeros_initializer)
        ret = tf.einsum("BFND,NDH->BFH", input_tensor, w)
        ret += b
    if activation is not None:
        return activation(ret)
    else:
        return ret


def abc_ced_abde(input_tensor,
                 num_attention_heads,
                 head_size,
                 initializer,
                 activation,
                 name=None):
    """A dense layer with 3D kernel.

  Args:
    input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
    num_attention_heads: Number of attention heads.
    head_size: The size per attention head.
    initializer: Kernel initializer.
    activation: Actication function.
    use_einsum: bool. Whether to use einsum or reshape+matmul for dense layers.
    name: The name scope of this layer.

  Returns:
    float logits Tensor.
  """

    input_shape = get_shape_list(input_tensor)
    hidden_size = input_shape[2]

    with tf.variable_scope(name):
        w = tf.get_variable(
            name="kernel",
            shape=[hidden_size, num_attention_heads * head_size],
            initializer=initializer)
        w = tf.reshape(w, [hidden_size, num_attention_heads, head_size])
        b = tf.get_variable(
            name="bias",
            shape=[num_attention_heads * head_size],
            initializer=tf.zeros_initializer)
        b = tf.reshape(b, [num_attention_heads, head_size])
        ret = tf.einsum("BFH,HND->BFND", input_tensor, w)
        ret += b
    if activation is not None:
        return activation(ret)
    else:
        return ret


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return contrib_layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def gelu(x):
    """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert tensor.shape.ndims in expected_rank, \
            "tensor {} shape {} is not equal expected rank {}".format(
                name, tensor.shape.ndims, expected_rank)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape
