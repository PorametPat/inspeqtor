import warnings
import typing
import functools

FnInput = typing.ParamSpec("FnInput")
FnOutput = typing.TypeVar("FnOutput")


def warn_not_tested_function(
    fun: typing.Callable[FnInput, FnOutput],
) -> typing.Callable[FnInput, FnOutput]:
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


def deprecated(
    fun: typing.Callable[FnInput, FnOutput],
    replacement: typing.Optional[typing.Callable[FnInput, FnOutput]] = None,
    message: typing.Optional[str] = None,
) -> typing.Callable[FnInput, FnOutput]:
    """A decorator to mark a function as deprecated.

    Example usage:
    >>> @deprecated
    ... def f(a, b):
    ...   return a + b

    Args:
      fun: the deprecated function.
      replacement: the replacement function.

    Returns:
      the wrapped function.
    """
    if hasattr(fun, "__name__"):
        warning_message = f"The function {fun.__name__} is deprecated."
    else:
        warning_message = "The function is deprecated."

    if replacement is not None:
        warning_message += f" Use {replacement.__name__} instead."

    if message is not None:
        warning_message += message

    @functools.wraps(fun)
    def new_fun(*args, **kwargs):
        warnings.warn(warning_message, category=DeprecationWarning, stacklevel=2)
        return fun(*args, **kwargs)

    return new_fun
