from deprecated import deprecated
import jax
import jax.numpy as jnp
import typing
from dataclasses import dataclass, asdict, field
import pathlib
from flax.traverse_util import flatten_dict, unflatten_dict

from ..v1.control import sample_params, BaseControl
from ..v1.data import save_pytree_to_json, load_pytree_from_json


ParametersDictType = dict[str, typing.Union[float, jnp.ndarray]]
# ParametersDictType = typing.Dict[str, typing.Union['ParametersDictType', float, jnp.ndarray]]


@dataclass
class ControlSequence:
    """Control sequence, expect to be sum of atomic control."""

    controls: dict[str, BaseControl]
    total_dt: int
    structure: typing.Sequence[typing.Sequence[str]] | None = field(default=None)

    def __post_init__(self):
        # Cache the bounds
        self.lower, self.upper = self.get_bounds()
        # Create the order

        if self.structure is None:
            self.auto_order = []
            for ctrl_key in self.controls.keys():
                sub_control = []
                for ctrl_param_key in self.lower[ctrl_key].keys():
                    sub_control.append((ctrl_key, ctrl_param_key))

                self.auto_order += sub_control
            self.structure = self.auto_order
        else:
            self.auto_order = self.structure

    def get_structure(self) -> typing.Sequence[typing.Sequence[str]]:
        return self.auto_order

    def sample_params(self, key: jax.Array) -> dict[str, ParametersDictType]:
        """Sample control parameter

        Args:
            key (jax.Array): Random key

        Returns:
            dict[str, ParametersDictType]: control parameters
        """
        params_dict: dict[str, ParametersDictType] = {}
        for idx, ctrl_key in enumerate(self.controls.keys()):
            subkey = jax.random.fold_in(key, idx)
            params = sample_params(subkey, self.lower[ctrl_key], self.upper[ctrl_key])
            params_dict[ctrl_key] = params

        return params_dict

    def sample_params_v2(self, key: jax.Array) -> dict[str, ParametersDictType]:
        """Sample control parameter

        Args:
            key (jax.Array): Random key

        Returns:
            dict[str, ParametersDictType]: control parameters
        """
        return nested_sample(key, merge_lower_upper(self.lower, self.upper))

    def get_bounds(
        self,
    ) -> tuple[dict[str, ParametersDictType], dict[str, ParametersDictType]]:
        """Get the bounds of the controls

        Returns:
            tuple[list[ParametersDictType], list[ParametersDictType]]: tuple of list of lower and upper bounds.
        """

        lower_bounds = jax.tree.map(
            lambda x: x.get_bounds()[0],
            self.controls,
            is_leaf=lambda x: isinstance(x, BaseControl),
        )
        upper_bounds = jax.tree.map(
            lambda x: x.get_bounds()[1],
            self.controls,
            is_leaf=lambda x: isinstance(x, BaseControl),
        )

        return lower_bounds, upper_bounds

    def get_envelope(
        self, params_dict: dict[str, ParametersDictType]
    ) -> typing.Callable:
        """Create envelope function with given control parameters

        Args:
            params_list (dict[str, ParametersDictType]): control parameter to be used

        Returns:
            typing.Callable: Envelope function
        """
        callables = []
        for params_key, params_val in params_dict.items():
            callables.append(self.controls[params_key].get_envelope(params_val))

        # Create a function that returns the sum of the envelopes
        def envelope(t):
            return sum([c(t) for c in callables])

        return envelope

    def to_dict(self) -> dict[str, str | dict[str, str | float]]:
        """Convert self to dict

        Returns:
            dict[str, str | dict[str, str | float]]: dict contain argument necessary for re-initialization.
        """
        return {
            **asdict(self),
            "classname": {k: v.__class__.__name__ for k, v in self.controls.items()},
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, str | dict[str, str | float]],
        controls: dict[str, type[BaseControl]],
    ) -> "ControlSequence":
        """Construct self with the provided dictionary

        Args:
            data (dict[str, str  |  dict[str, str  |  float]]): The dictionary contain initialization arguments
            controls (dict[str, type[BaseControl]]): The map of control name and class of the control

        Returns:
            ControlSequence: the instance of control sequence
        """
        controls_data = data["controls"]
        assert isinstance(controls_data, dict)

        instantiated_controls = {}

        for (ctrl_key, ctrl_data), (ctrl_key_match, ctrl_cls) in zip(
            controls_data.items(), controls.items()
        ):
            assert ctrl_key == ctrl_key_match
            assert isinstance(ctrl_data, dict), f"Expected dict, got {type(ctrl_data)}"
            instantiated_controls[ctrl_key] = ctrl_cls.from_dict(ctrl_data)

        total_dt = data["total_dt"]
        assert isinstance(total_dt, int)
        structure = data["structure"]
        assert isinstance(structure, list)
        # Explicitly convert each item in the structure to be tuple.
        structure = [tuple(item) for item in structure]

        return cls(
            controls=instantiated_controls, total_dt=total_dt, structure=structure
        )

    def to_file(self, path: typing.Union[str, pathlib.Path]):
        """Save configuration of the pulse to file given folder path.

        Args:
            path (typing.Union[str, pathlib.Path]): Path to the folder to save sequence, will be created if not existed.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)

        save_pytree_to_json(self.to_dict(), path / "control_sequence.json")

    @classmethod
    def from_file(
        cls,
        path: typing.Union[str, pathlib.Path],
        controls: dict[str, type[BaseControl]],
    ):
        """Initialize itself from a file.

        Args:
            path (typing.Union[str, pathlib.Path]): Path to file.
            controls (dict[str, type[BaseControl]]): The map of control name and class of the control

        Returns:
            ControlSequence: the instance of control sequence
        """
        if isinstance(path, str):
            path = pathlib.Path(path)

        ctrl_loaded_dict = load_pytree_from_json(
            path / "control_sequence.json", lambda k, v: (True, v)
        )

        return cls.from_dict(ctrl_loaded_dict, controls=controls)


def control_waveform(
    param: ParametersDictType,
    t_eval: jnp.ndarray,
    control: BaseControl,
) -> jnp.ndarray:
    return jax.vmap(control.get_envelope(param))(t_eval)


def sequence_waveform(
    params: dict[str, ParametersDictType],
    t_eval: jnp.ndarray,
    control_seqeunce: ControlSequence,
) -> jnp.ndarray:
    """
    Samples the pulse sequence by generating random parameters for each pulse and computing the total waveform.

    Parameters:
        key (Key): The random key used for generating the parameters.

    Returns:
        tuple[list[ParametersDictType], Complex[Array, "time"]]: A tuple containing a list of parameter dictionaries for each pulse and the total waveform.

    Example:
        key = jax.random.PRNGKey(0)
        params, total_waveform = sample(key)
    """
    # Create base waveform
    total_waveform = jnp.zeros_like(t_eval, dtype=jnp.complex64)

    for (param_key, param_val), (ctrl_key, control) in zip(
        params.items(), control_seqeunce.controls.items()
    ):
        waveform = control_waveform(param_val, t_eval, control)
        total_waveform += waveform

    return total_waveform


def merge_lower_upper(lower, upper):
    """Merge lower and upper bound into bounds

    Args:
        lower (_type_): The lower bound
        upper (_type_): The upper bound

    Returns:
        _type_: Bound from the lower and upper.
    """
    return jax.tree.map(lambda x, y: (x, y), lower, upper)


def split_bounds(bounds):
    """Create lower and upper bound from bounds

    Args:
        bounds (_type_): The bounds to extract the lower and upper bound

    Returns:
        _type_: The lower and upper bound
    """
    return jax.tree.map(
        lambda x: x[0], bounds, is_leaf=lambda x: isinstance(x, tuple)
    ), jax.tree.map(lambda x: x[1], bounds, is_leaf=lambda x: isinstance(x, tuple))


def uniform_sample(key: jnp.ndarray, bound: tuple[float, float]):
    return jax.random.uniform(key, minval=bound[0], maxval=bound[1])


def nested_sample(key: jnp.ndarray, bounds, sample_fn=uniform_sample):
    """Sample from nested bounds with custom sampling function `sample_fn`

    Args:
        key (jnp.ndarray): Random key
        bounds (_type_): Bound of the control parameter
        sample_fn (_type_, optional): Custom sampling function. Defaults to uniform_sample.

    Returns:
        _type_: Control parameter sample from bound
    """
    return unflatten_dict(
        {
            k: sample_fn(jax.random.fold_in(key, idx), bound)
            for idx, (k, bound) in enumerate(flatten_dict(bounds).items())
        }
    )


def check_bounds(param, bounds) -> bool:
    """Check if the given control parameter violate the bound or not.

    Args:
        param (_type_): Control parameter
        bounds (_type_): Bound of control parameter

    Returns:
        bool: `True` if parameter do not violate the bound, otherwise `False`
    """
    valid_container = jax.tree.map(
        lambda x, bound: (bound[0] < x) & (x < bound[1]), param, bounds
    )
    return jax.tree.reduce(lambda init, x: init & x, valid_container, initializer=True)


def get_value_by_keys(param, dict_keys):
    return jax.tree.reduce(lambda init, x: init[x], dict_keys, initializer=param)


def ravel_unravel_fn(structure: typing.Iterable[typing.Iterable[str]]):
    """This function return the ravel and unravel functions for the provided control sequence

    Args:
        structure (typing.Iterable[typing.Iterable[str]]): The structure of the pytree

    Returns:
        tuple[typing.Callable, typing.Callable]: The first element is the function that convert structured parameter to array, the second is a function that reverse the action of the first.
    """

    def ravel_fn(param):
        return jnp.array(
            [get_value_by_keys(param, dict_keys) for dict_keys in structure]
        )

    def unravel_fn(param: jnp.ndarray):
        return unflatten_dict(
            {dict_keys: param[idx] for idx, dict_keys in enumerate(structure)}
        )

    return ravel_fn, unravel_fn


@deprecated(reason="Old implementation of the ravel_unravel_fn")
def ravel_unravel_fn_old(control_sequence: ControlSequence):
    """This function return the ravel and unravel functions for the provided control sequence

    Args:
        control_sequence (ControlSequence): The control sequence

    Returns:
        tuple[typing.Callable, typing.Callable]: The first element is the function that convert structured parameter to array, the second is a function that reverse the action of the first.
    """
    structure = control_sequence.get_structure()

    def ravel_fn(param_dict: dict[str, ParametersDictType]) -> jnp.ndarray:
        tmp = []
        for sub_order in structure:
            tmp.append(param_dict[sub_order[0]][sub_order[1]])
        return jnp.array(tmp)

    def unravel_fn(param: jnp.ndarray) -> dict[str, ParametersDictType]:
        tmp = {}
        for idx, sub_order in enumerate(structure):
            if sub_order[0] not in tmp:
                tmp[sub_order[0]] = {}

            tmp[sub_order[0]][sub_order[1]] = param[idx]
        return tmp

    return ravel_fn, unravel_fn


def construct_control_sequence_reader(
    controls: list[type[BaseControl]] = [],
) -> typing.Callable[[typing.Union[str, pathlib.Path]], ControlSequence]:
    """Construct the control sequence reader

    Args:
        controls (list[type[BasePulse]], optional): List of control constructor. Defaults to [].

    Returns:
        typing.Callable[[typing.Union[str, pathlib.Path]], controlsequence]: Control sequence reader that will automatically contruct control sequence from path.
    """
    default_controls: list[type[BaseControl]] = []

    # Merge the default controls with the provided controls
    controls_list = default_controls + controls
    control_dict = {ctrl.__name__: ctrl for ctrl in controls_list}

    def control_sequence_reader(
        path: typing.Union[str, pathlib.Path],
    ) -> ControlSequence:
        """Construct control sequence from path

        Args:
            path (typing.Union[str, pathlib.Path]): Path of the saved control sequence configuration.

        Returns:
            ControlSeqence: Control sequence instance.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)

        control_sequence_dict = load_pytree_from_json(
            path / "control_sequence.json", lambda k, v: (True, v)
        )

        parsed_controls = {}
        assert isinstance(control_sequence_dict["classname"], dict)

        for ctrl_key, ctrl_classname in control_sequence_dict["classname"].items():
            parsed_controls[ctrl_key] = control_dict[ctrl_classname]

        return ControlSequence.from_dict(
            control_sequence_dict, controls=parsed_controls
        )

    return control_sequence_reader


def get_envelope_transformer(control_sequence: ControlSequence):
    """Generate get_envelope function with control parameter array as an input instead of list form

    Args:
        control_sequence (ControlSequence): Control seqence instance

    Returns:
        typing.Callable[[jnp.ndarray], typing.Any]: Transformed get envelope function
    """
    _, unravel_fn = ravel_unravel_fn(control_sequence.get_structure())

    def get_envelope(params: jnp.ndarray) -> typing.Callable[..., typing.Any]:
        return control_sequence.get_envelope(unravel_fn(params))

    return get_envelope


def ravel_transform(
    fn: typing.Callable, control_sequence: ControlSequence, at: int = 0
) -> typing.Callable:
    """Transform the argument at index `at` of the function `fn` with `unravel_fn` of the control sequence

    Note:
        ```python
        signal_fn = sq.control.ravel_transform(
            sq.physics.signal_func_v5(control_sequence.get_envelope, qubit_info.frequency, dt),
            control_sequence,
        )
        ```

    Args:
        fn (typing.Callable): The function to be transformed
        control_sequence (ControlSequence): The control sequence that will use to produce `unravel_fn`.

    Returns:
        typing.Callable: A function that its first argument is transformed by `unravel_fn`
    """
    _, unravel_fn = ravel_unravel_fn(control_sequence.get_structure())

    def wrapper(*args, **kwargs):
        list_args = list(args)
        list_args[at] = unravel_fn(list_args[at])

        return fn(*tuple(list_args), **kwargs)

    return wrapper


def get_envelope(param, seq: ControlSequence):
    """Return an envelope function create from envelope of all controls in `seq` with control parameter `param`

    Args:
        param (_type_): Control parameter
        seq (ControlSequence): Control Sequence

    Returns:
        _type_: A function of time which is a sum of all envelope of control in `seq` with parameter `param`
    """
    tree = jax.tree.map(
        lambda ctrl, x: ctrl.get_envelope(x),
        seq.controls,
        param,
        is_leaf=lambda x: isinstance(x, BaseControl),
    )

    def envelope(t):
        return jax.tree.reduce(lambda value, fn: fn(t) + value, tree, initializer=0.0)

    return envelope


def envelope_fn(param, t: jnp.ndarray, seq: ControlSequence):
    """Return an envelope of all of the control in control sequence `seq` given paramter `param` at time `t`

    Args:
        param (_type_): Control parameter
        t (jnp.ndarray): Time to evaluate the envelope
        seq (ControlSequence): The control sequence to get the envelope

    Returns:
        _type_: Envelope of all control in `seq` evaluate at time `t` with parameter `param`
    """
    tree = jax.tree.map(
        lambda ctrl, x: ctrl.get_envelope(x)(t),
        seq.controls,
        param,
        is_leaf=lambda x: isinstance(x, BaseControl),
    )

    return jax.tree.reduce(jnp.add, tree)
