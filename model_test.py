import random

import tensorflow as tf
import modeling


class ModelTest(tf.test.TestCase):
    class ModelTester(object):
        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_input_mask=True,
                     use_token_type_ids=True,
                     vocab_size=99,
                     embedding_size=32,
                     hidden_size=32,
                     num_hidden_layers=6,
                     num_hidden_groups=2,
                     num_attention_heads=4,
                     intermediate_size=37,
                     hidden_act="gelu",
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     max_position_embeddings=512,
                     type_vocab_size=16,
                     initializer_range=0.02,
                     scope=None):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_mask = use_input_mask
            self.use_token_type_ids = use_token_type_ids
            self.vocab_size = vocab_size
            self.embedding_size = embedding_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_hidden_groups = num_hidden_groups
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.scope = scope

        def create_model(self):
            input_ids = ModelTest.ids_tensor([self.batch_size, self.seq_length],
                                             self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = ModelTest.ids_tensor(
                    [self.batch_size, self.seq_length], vocab_size=2)

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = ModelTest.ids_tensor(
                    [self.batch_size, self.seq_length], self.type_vocab_size)

            config = modeling.ModelConfig(
                vocab_size=self.vocab_size,
                embedding_size=self.embedding_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_hidden_groups=self.num_hidden_groups,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                type_vocab_size=self.type_vocab_size,
                initializer_range=self.initializer_range)

            model = modeling.Model(
                config=config,
                is_training=self.is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
                scope=self.scope)

            outputs = {
                "embedding_output": model.get_embedding_output(),
                "sequence_output": model.get_sequence_output(),
                "pooled_output": model.get_pooled_output(),
                "all_encoder_layers": model.get_all_encoder_layers(),
            }
            return outputs


    def test_default(self):
        self.run_tester(ModelTest.ModelTester(self))

    def run_tester(self, tester):
        with self.test_session() as sess:
            ops = tester.create_model()
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)

            output_result = sess.run(ops)
            writer = tf.summary.FileWriter("test", sess.graph)

            writer.close()

            # self.assert_all_tensors_reachable(sess, [init_op, ops])

    @classmethod
    def ids_tensor(cls, shape, vocab_size, rng=None, name=None):
        """Creates a random int32 tensor of the shape within the vocab size."""
        if rng is None:
            rng = random.Random()

        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(rng.randint(0, vocab_size - 1))

        return tf.constant(value=values, dtype=tf.int32, shape=shape, name=name)


if __name__ == "__main__":
    tf.test.main()
