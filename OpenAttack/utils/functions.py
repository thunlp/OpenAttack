from ..exceptions import UnknownParameterException


def check_parameters(keys, config):
    """
    Check config parameters.

    :raises UnknownParameterException: For any parameter not in keys.
    """
    for key in config.keys():
        if key not in keys:
            raise UnknownParameterException(key)
    return
