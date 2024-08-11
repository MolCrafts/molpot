from functools import wraps


def loss_wrapper(input_: str, target: str):

    def decorate(func):

        @wraps(func)
        def wrapper(inputs):
            return func(inputs[input_], inputs[target])

        return wrapper

    return decorate
