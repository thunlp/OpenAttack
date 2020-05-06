from ..exceptions import UnknownParameterException


def check_parameters(keys, config):
    for key in config.keys():
        if key not in keys:
            raise UnknownParameterException
    return
