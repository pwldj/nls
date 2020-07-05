import multiprocessing
import os
import random
import threading

import numpy as np
import six
import tensorflow as tf

import tokenization
import collections
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_integer(
    "writer_num", 10,
    ""
)

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_shuffle", True,
    "Whether do whole word shuffle"
)

flags.DEFINE_bool(
    "shuffle_masked", False,
    ""
)

flags.DEFINE_bool(
    "do_whole_word_mask", True,
    "Whether do whole word mask"
)

flags.DEFINE_integer(
    "max_mask_token", 20,
    ""
)

flags.DEFINE_integer(
    "max_shuffle_time", 5,
    ""
)

flags.DEFINE_float(
    "max_mask_prob", 0.2,
    ""
)

flags.DEFINE_integer(
    "dupe_factor", 40,
    ""
)

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    ""
)

flags.DEFINE_integer(
    "num_docs_pre_yield", 5000,
    ""
)

flags.DEFINE_integer(
    "num_thread", 16,
    ""
)

flags.DEFINE_integer(
    "max_seq_length", 512,
    ""
)

rng = random.Random()


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, shuffle_index, is_random_next,
                 length):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.shuffle_index = shuffle_index
        self.length = length

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "shuffle_index: %s\n" % (" ".join(
            [str(x) for x in self.shuffle_index]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def merge_tokens(tokens_c, masked_lm_positions_c, masked_lm_labels_c, shuffle_index_c, sep):
    tokens = []
    masked_lm_positions = []
    masked_lm_labels = []
    shuffle_index = []
    segment_ids = []
    curr_t = 0

    if rng.random() < 0.5:
        is_random_next = True
        tokens_c = tokens_c[sep:] + tokens_c[:sep]
        masked_lm_positions_c = masked_lm_positions_c[sep:] + masked_lm_positions_c[:sep]
        masked_lm_labels_c = masked_lm_labels_c[sep:] + masked_lm_labels_c[:sep]
        shuffle_index_c = shuffle_index_c[sep:] + shuffle_index_c[:sep]
        sep = len(tokens_c) - sep
    else:
        is_random_next = False

    tokens.append("[CLS]")
    segment_ids.append(0)
    shuffle_index.append(curr_t)
    curr_t += 1

    for i in range(sep):
        tokens.extend(tokens_c[i])
        masked_lm_positions.extend([x + curr_t for x in masked_lm_positions_c[i]])
        masked_lm_labels.extend(masked_lm_labels_c[i])
        shuffle_index.extend([x + curr_t for x in shuffle_index_c[i]])
        segment_ids.extend([0] * len(tokens_c[i]))
        curr_t += len(tokens_c[i])

    tokens.append("[SEP]")
    segment_ids.append(0)
    shuffle_index.append(curr_t)
    curr_t += 1

    for i in range(sep, len(tokens_c)):
        tokens.extend(tokens_c[i])
        masked_lm_positions.extend([x + curr_t for x in masked_lm_positions_c[i]])
        masked_lm_labels.extend(masked_lm_labels_c[i])
        shuffle_index.extend([x + curr_t for x in shuffle_index_c[i]])
        segment_ids.extend([1] * len(tokens_c[i]))
        curr_t += len(tokens_c[i])

    tokens.append("[SEP]")
    segment_ids.append(1)
    shuffle_index.append(curr_t)
    curr_t += 1

    instance = TrainingInstance(tokens, segment_ids, masked_lm_positions, masked_lm_labels, shuffle_index,
                                is_random_next, curr_t)

    return instance


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (sentence piece)."""
    special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
    special_pieces.add(u"€".encode("utf-8"))
    special_pieces.add(u"£".encode("utf-8"))
    # Note(mingdachen):
    # For foreign characters, we always treat them as a whole piece.
    english_chars = set(list("abcdefghijklmnopqrstuvwxyz"))
    if (six.ensure_str(piece).startswith("▁") or
            six.ensure_str(piece).startswith("<") or piece in special_pieces or piece in ["[CLS]", "[SEP]", "[MASK]"] or
            not all([i.lower() in english_chars.union(special_pieces)
                     for i in piece])):
        return True
    else:
        return False


def create_mask_lm(tokens, max_mask_pre_seq):
    if max_mask_pre_seq <= 0 or rng.random() < 0.52 or not tokens:
        return tokens, [], []
    tokens = [x for x in tokens]

    tokens_len = len(tokens)
    max_mask_token = min(max_mask_pre_seq, int(FLAGS.max_mask_prob * tokens_len) + 1)

    index = []
    if FLAGS.do_whole_word_mask:
        for i in range(tokens_len):
            if is_start_piece(tokens[i]) or not index:
                index.append([i])
            else:
                index[-1].append(i)
    else:
        index = [[x] for x in range(tokens_len)]

    ngram = []
    i = 0
    while i < len(index):
        package = []
        n = np.random.choice([1, 2, 3], p=[6 / 11, 3 / 11, 2 / 11])
        for _ in range(min(n, len(index) - i)):
            package.append(index[i])
            i += 1
        ngram.append(package)

    masked_lm_positions = []
    masked_lm_labels = []
    rng.shuffle(ngram)
    for n in ngram:
        flat = sum(n, [])
        if len(flat) + len(masked_lm_labels) <= max_mask_token:
            for f in flat:
                masked_lm_labels.append(tokens[f])
                masked_lm_positions.append(f)
                tokens[f] = "[MASK]"
        if len(masked_lm_labels) >= max_mask_token:
            break
    return tokens, masked_lm_positions, masked_lm_labels


def shuffle_token(tokens):
    if not tokens:
        return tokens, [x for x in range(len(tokens))]

    index = []
    words = []
    if FLAGS.do_whole_word_shuffle:
        for i, t in enumerate(tokens):
            if is_start_piece(t) or not words:
                words.append([t])
                index.append([i])
            else:
                words[-1].append(t)
                index[-1].append(i)
    else:
        words = [[t] for t in tokens]
        index = [[x] for x in range(len(tokens))]

    if FLAGS.shuffle_masked:
        z = zip(words, index)
        z = rng.shuffle(z)
        words, index = zip(*z)
        return sum(words, []), sum(index, [])

    for i in range(len(words) - 1):
        if words[i] == ["[MASK]"]:
            continue
        random_next = None
        for _ in range(FLAGS.max_shuffle_time):
            random_next = rng.randint(i + 1, len(words) - 1)
            if words[random_next] != ["[MASK]"] and len(words[i]) == len(words[random_next]):
                words[i], words[random_next] = words[random_next], words[i]
                index[i], index[random_next] = index[random_next], index[i]
                break

    return sum(words, []), sum(index, [])


def truncate_seq_pair(chunk, sep, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""

    a_chunk = chunk[:sep]
    b_chunk = chunk[sep:]
    while True:
        a_length = sum([len(x) for x in a_chunk])
        b_length = sum([len(x) for x in b_chunk])
        total_length = a_length + b_length
        if total_length <= max_num_tokens:
            break

        reduce_chunk = a_chunk if a_length > b_length else b_chunk
        if rng.random() < 0.5:
            del reduce_chunk[0][0]
            if not reduce_chunk[0]:
                del reduce_chunk[0]
        else:
            del reduce_chunk[-1][-1]
            if not reduce_chunk[-1]:
                del reduce_chunk[-1]

    return a_chunk + b_chunk, len(a_chunk)


def create_instances(data):
    # index = data[0]
    # docs = data[1]
    # doc = docs[index]
    doc = data

    max_num_tokens = FLAGS.max_seq_length - 3

    i = 0
    instances = []
    chunk = []
    tokens_len = 0
    while i < len(doc):
        if tokens_len < max_num_tokens:
            chunk.append(doc[i])
            tokens_len += len(doc[i])
            i += 1
            continue

        if chunk:
            if rng.random() < FLAGS.short_seq_prob:
                s = rng.randint(1, len(chunk))
                if sum([len(x) for x in chunk[:s]]) < 10:
                    continue
                i -= (len(chunk) - s)
                chunk = chunk[:s]
            if len(chunk) == 1:
                s = rng.randint(1, len(chunk[0]) - 1)
                chunk = [chunk[0][:s], chunk[0][s:]]
            assert len(chunk) > 1

            sep = rng.randint(1, len(chunk) - 1)
            chunk, sep = truncate_seq_pair(chunk, sep, max_num_tokens)

            tokens = []
            masked_lm_labels = []
            masked_lm_positions = []
            shuffle_index = []
            tokens_len = 0
            max_mask = FLAGS.max_mask_token
            pre_seq_mask = int((max_mask / len(chunk) + 1) * 2)
            for c in chunk:
                num_mask = min(max_mask, rng.randint(0, pre_seq_mask))
                if FLAGS.shuffle_masked:
                    tok, shuf = shuffle_token(c)
                    tok, pos, label = create_mask_lm(tok, num_mask)
                else:
                    tok, pos, label = create_mask_lm(c, num_mask)
                    tok, shuf = shuffle_token(tok)

                tokens.append(tok)
                max_mask -= len(label)
                masked_lm_labels.append(label)
                masked_lm_positions.append(pos)
                shuffle_index.append(shuf)
                tokens_len += len(tok)

            instances.append(merge_tokens(tokens, masked_lm_positions, masked_lm_labels, shuffle_index, sep))

            chunk = []
            tokens_len = 0
            i += 1

    return instances


def get_docs(input_files, num_docs, tokenizer):
    docs = [[]]

    for file in input_files:
        tf.logging.info("Reading file:{}".format(file))
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if FLAGS.spm_model_file:
                    line = tokenization.preprocess_text(line, lower=FLAGS.do_lower_case)
                else:
                    line = tokenization.convert_to_unicode(line).strip()
                if line and not line.startswith('#'):
                    tokens = tokenizer.tokenize(line)
                    docs[-1].append(tokens)
                elif docs[-1]:
                    if num_docs != 0 and len(docs) == num_docs:
                        yield docs
                        docs = [[]]
                    else:
                        docs.append([])
            yield docs
            docs = [[]]


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def write_instance(instances, writer, max_seq_length, tokenizer):
    for instance in instances:
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        shuffle_index = list(instance.shuffle_index)
        assert len(input_ids) <= max_seq_length

        i = len(input_ids)
        while i < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            shuffle_index.append(i)
            i += 1

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(shuffle_index) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        assert len(masked_lm_positions) <= FLAGS.max_mask_token

        while len(masked_lm_positions) < FLAGS.max_mask_token:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        assert len(masked_lm_positions) == FLAGS.max_mask_token
        assert len(masked_lm_ids) == FLAGS.max_mask_token
        assert len(masked_lm_weights) == FLAGS.max_mask_token

        sentence_order_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["shuffle_index"] = create_int_feature(shuffle_index)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        # Note: We keep this feature name `next_sentence_labels` to be compatible
        # with the original data created by lanzhzh@. However, in the ALBERT case
        # it does contain sentence_order_label.
        features["next_sentence_labels"] = create_int_feature(
            [sentence_order_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(tf_example.SerializeToString())

        # if rng.random() < 0.001:
        #     for feature_name in features.keys():
        #         feature = features[feature_name]
        #         values = []
        #         if feature.int64_list.value:
        #             values = feature.int64_list.value
        #         elif feature.float_list.value:
        #             values = feature.float_list.value
        #         tf.logging.info(
        #             "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    return len(instances)


def main(_):
    input_files = os.listdir(FLAGS.input_file)
    input_files = [os.path.join(FLAGS.input_file, f) for f in input_files]

    pool = multiprocessing.Pool(FLAGS.num_thread)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
        spm_model_file=FLAGS.spm_model_file)

    writers = []
    for i in range(FLAGS.writer_num):
        writers.append(tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_file, "example_{}.tfrecord".format(i))))
    writers_id = 0

    count = 0
    for docs in get_docs(input_files, FLAGS.num_docs_pre_yield, tokenizer):
        rng.shuffle(docs)
        # docs = [(i, docs) for i in range(len(docs))]
        for _ in range(FLAGS.dupe_factor):
            for instances in pool.imap(create_instances, docs):
                n = write_instance(instances, writers[writers_id], FLAGS.max_seq_length, tokenizer)
                writers_id = (writers_id + 1) % len(writers)
                count += n

            # for d in docs:
            #     instances = create_instances(d)
            #     n = write_instance(instances, writers[writers_id], FLAGS.max_seq_length, tokenizer)
            #     writers_id = (writers_id + 1) % len(writers)
            #     count += n

            tf.logging.info("already write {} instances".format(count))


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
