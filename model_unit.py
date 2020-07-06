import collections
import os
import re

import tensorflow as tf


def init_from_checkpoint(FLAGS, global_vars=False):
    tvars = tf.global_variables() if global_vars else tf.trainable_variables()
    initialized_variable_names = {}
    if FLAGS.init_checkpoint is not None:
        if FLAGS.init_checkpoint.endswith("latest"):
            ckpt_dir = os.path.dirname(FLAGS.init_checkpoint)
            init_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        else:
            init_checkpoint = FLAGS.init_checkpoint

        tf.logging.info("Initialize from the ckpt {}".format(init_checkpoint))

        (assignment_map, initialized_variable_names
         ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # Log customized initialization
        tf.logging.info("**** Global Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        # tf.logging.info('original name: %s', name)
        if name not in name_to_variable:
            continue
        # assignment_map[name] = name
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names
