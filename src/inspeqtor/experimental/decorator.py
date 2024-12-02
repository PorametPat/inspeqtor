import warnings
import typing
import functools


FnOutput = typing.TypeVar("FnOutput")


def warn_not_tested_function(
    fun: typing.Callable[..., FnOutput],
) -> typing.Callable[..., FnOutput]:
    """A decorator to mark a function not yet tested.

    Example usage:
    >>> @warn_not_tested_function
    ... def f(a, b):
    ...   return a + b

    Args:
      fun: the deprecated function.

    Returns:
      the wrapped function.
    """
    if hasattr(fun, "__name__"):
        warning_message = f"The function {fun.__name__} is not yet tested."
    else:
        warning_message = "The function is not yet tested."

    @functools.wraps(fun)
    def new_fun(*args, **kwargs):
        warnings.warn(warning_message, category=UserWarning, stacklevel=2)
        return fun(*args, **kwargs)

    return new_fun
