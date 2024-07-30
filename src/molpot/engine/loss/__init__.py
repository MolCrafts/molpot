from functools import wraps


def loss_wrapper(input_: str, target: str):

    def decorate(func):

        @wraps(func)
        def wrapper(inputs, outputs):
            return func(inputs[input_], outputs[target])

        return wrapper

    return decorate
