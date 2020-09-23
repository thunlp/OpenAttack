

def init_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        return
    import os
    import logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel(logging.ERROR)
    return