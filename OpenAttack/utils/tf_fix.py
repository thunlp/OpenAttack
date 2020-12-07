

def init_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        return
    import os
    import logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel(logging.ERROR)

    # gpus = tf.config.list_physical_devices('GPU')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return