import jax
import jax.numpy as jnp
import typing
from inspeqtor.experimental.ctyping import ParametersDictType
from dataclasses import dataclass, asdict, field
import pathlib

from ..experimental.control import BaseControl, sample_params
from ..experimental.data import save_pytree_to_json, load_pytree_from_json


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

        self.auto_order = []
        for ctrl_key in self.controls.keys():
            sub_control = []
            for ctrl_param_key in self.lower[ctrl_key].keys():
                sub_control.append((ctrl_key, ctrl_param_key))

            self.auto_order += sub_control

        self.structure = self.auto_order

    def get_structure(self):
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

    def get_bounds(
        self,
    ) -> tuple[dict[str, ParametersDictType], dict[str, ParametersDictType]]:
        """Get the bounds of the controls

        Returns:
            tuple[list[ParametersDictType], list[ParametersDictType]]: tuple of list of lower and upper bounds.
        """
        lower_bounds = {}
        upper_bounds = {}
        for key, pulse in self.controls.items():
            lower, upper = pulse.get_bounds()
            lower_bounds[key] = lower
            upper_bounds[key] = upper

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


def get_waveform(
    params: dict[str, ParametersDictType], control_seqeunce: ControlSequence
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
    total_waveform = jnp.zeros(control_seqeunce.total_dt, dtype=jnp.complex64)

    for (param_key, param_val), (ctrl_key, control) in zip(
        params.items(), control_seqeunce.controls.items()
    ):
        waveform = control.get_waveform(param_val)
        total_waveform += waveform

    return total_waveform


def ravel_unravel_fn(control_sequence: ControlSequence):
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
    _, unravel_fn = ravel_unravel_fn(control_sequence)

    def get_envelope(params: jnp.ndarray) -> typing.Callable[..., typing.Any]:
        return control_sequence.get_envelope(unravel_fn(params))

    return get_envelope
