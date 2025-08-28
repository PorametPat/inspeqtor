import os
import jax
import jax.numpy as jnp
import typing
from dataclasses import dataclass, asdict
import json
from abc import ABC, abstractmethod
import pathlib

from .ctyping import ParametersDictType


def sample_params(
    key: jnp.ndarray, lower: ParametersDictType, upper: ParametersDictType
) -> ParametersDictType:
    """Sample parameters with the same shape with given lower and upper bounds

    Args:
        key (jnp.ndarray): Random key
        lower (ParametersDictType): Lower bound
        upper (ParametersDictType): Upper bound

    Returns:
        ParametersDictType: Dict of the sampled parameters
    """
    # This function is general because it is depend only on lower and upper structure
    param: ParametersDictType = {}
    param_names = lower.keys()
    for name in param_names:
        sample_key, key = jax.random.split(key)
        param[name] = jax.random.uniform(
            sample_key, shape=(), dtype=float, minval=lower[name], maxval=upper[name]
        )

    # return jax.tree.map(float, param)
    return param


@dataclass
class BaseControl(ABC):
    duration: int

    def __post_init__(self):
        self.t_eval = jnp.arange(0, self.duration, 1)
        self.validate()

    def validate(self):
        # Validate that all attributes are json serializable
        try:
            json.dumps(self.to_dict())
        except TypeError as e:
            raise TypeError(
                f"Cannot serialize {self.__class__.__name__} to json"
            ) from e

        lower, upper = self.get_bounds()
        # Validate that the sampling function is working
        key = jax.random.PRNGKey(0)
        params = sample_params(key, lower, upper)
        waveform = self.get_waveform(params)

        assert all(
            [isinstance(k, str) for k in params.keys()]
        ), "All key of params dict must be string"
        assert all(
            [isinstance(v, float) for v in params.values()]
        ), "All value of params dict must be float"
        assert isinstance(waveform, jax.Array), "Waveform must be jax.Array"

        # Validate that params is serializable and deserializable
        try:
            reread_params = json.loads(json.dumps(params))
            assert params == reread_params

        except TypeError as e:
            raise TypeError(
                f"Cannot serialize params dict of {self.__class__.__name__} to json"
            ) from e

    @abstractmethod
    def get_bounds(
        self, *arg, **kwarg
    ) -> tuple[ParametersDictType, ParametersDictType]: ...

    @abstractmethod
    def get_envelope(self, params: ParametersDictType) -> typing.Callable:
        raise NotImplementedError("get_envelopes method is not implemented")

    def get_waveform(self, params: ParametersDictType) -> jnp.ndarray:
        """Get the discrete waveform of the pulse

        Args:
            params (ParametersDictType): Control parameter

        Returns:
            jnp.ndarray: Waveform of the control.
        """
        return jax.vmap(self.get_envelope(params), in_axes=(0,))((self.t_eval))

    def to_dict(self) -> dict[str, typing.Union[int, float, str]]:
        """Convert the control configuration to dictionary

        Returns:
            dict[str, typing.Union[int, float, str]]: Configuration of the control
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Construct the control instace from the dictionary.

        Args:
            data (dict): Dictionary for construction of the control instance.

        Returns:
            The instance of the control.
        """
        return cls(**data)


@dataclass
class ControlSequence:
    """Control sequence, expect to be sum of atomic control."""

    controls: typing.Sequence[BaseControl]
    total_dt: int
    validate: bool = True

    def __post_init__(self):
        # validate that each pulse have len of total_dt
        if self.validate:
            self._validate()

    def _validate(self):
        # Must check that the sum of the pulse lengths is equal to the total length of the pulse sequence
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, self.total_dt)
        for pulse_key, pulse in zip(subkeys, self.controls):
            params = sample_params(pulse_key, *pulse.get_bounds())
            waveform = pulse.get_waveform(params)
            assert isinstance(waveform, jax.Array)
            # Assert the waveform is of the correct length
            assert waveform.shape == (self.total_dt,)
            # Assert that all key of params dict is string and all value is jax.Array
            assert all(
                [isinstance(k, str) for k in params.keys()]
            ), "All key of params dict must be string"
            assert all(
                [isinstance(v, (float, int, jnp.ndarray)) for v in params.values()]
            ), "All value of params dict must be float or jax.Array"

        params = self.sample_params(key)

        # Assert that the bounds have the same pytree structure as the parameters
        lower, upper = self.get_bounds()

        assert jax.tree.structure(lower) == jax.tree.structure(params)
        assert jax.tree.structure(upper) == jax.tree.structure(params)

    def sample_params(self, key: jax.Array) -> list[ParametersDictType]:
        """Sample control parameter

        Args:
            key (jax.Array): Random key

        Returns:
            list[ParametersDictType]: control parameters
        """
        # Split key for each pulse
        subkeys = jax.random.split(key, self.total_dt)

        params_list: list[ParametersDictType] = []
        for pulse_key, pulse in zip(subkeys, self.controls):
            params = sample_params(pulse_key, *pulse.get_bounds())
            params_list.append(params)

        return params_list

    def get_waveform(self, params_list: list[ParametersDictType]) -> jnp.ndarray:
        """
        Samples the pulse sequence by generating random parameters for each pulse and computing the total waveform.

        Parameters:
            key (Key): The random key used for generating the parameters.

        Returns:
            tuple[list[ParametersDictType], Complex[Array, "time"]]: A tuple containing a list of parameter dictionaries for each pulse and the total waveform.

        Example:
            key = jax.random.PRNGKey(0)
            params_list, total_waveform = sample(key)
        """
        # Create base waveform
        total_waveform = jnp.zeros(self.total_dt, dtype=jnp.complex64)

        for _params, _pulse in zip(params_list, self.controls):
            waveform = _pulse.get_waveform(_params)
            total_waveform += waveform

        return total_waveform

    def get_envelope(self, params_list: list[ParametersDictType]) -> typing.Callable:
        """Create envelope function with given control parameters

        Args:
            params_list (list[ParametersDictType]): control parameter to be used

        Returns:
            typing.Callable: Envelope function
        """
        callables = []
        for _params, _pulse in zip(params_list, self.controls):
            callables.append(_pulse.get_envelope(_params))

        # Create a function that returns the sum of the envelopes
        def envelope(t):
            return sum([c(t) for c in callables])

        return envelope

    def get_bounds(self) -> tuple[list[ParametersDictType], list[ParametersDictType]]:
        """Get the bounds of the controls

        Returns:
            tuple[list[ParametersDictType], list[ParametersDictType]]: tuple of list of lower and upper bounds.
        """
        lower_bounds = []
        upper_bounds = []
        for pulse in self.controls:
            lower, upper = pulse.get_bounds()
            lower_bounds.append(lower)
            upper_bounds.append(upper)

        return lower_bounds, upper_bounds

    def get_parameter_names(self) -> list[list[str]]:
        """Get the name of the control parameters in the control sequence.

        Returns:
            list[list[str]]: Structured name of control parameters.
        """
        # Sample the pulse sequence to get the parameter names
        key = jax.random.key(0)
        params_list = self.sample_params(key)

        # Get the parameter names for each pulse
        parameter_names = []
        for params in params_list:
            parameter_names.append(list(params.keys()))

        return parameter_names

    def to_dict(self) -> dict[str, typing.Any]:
        """Convert control sequence to dictionary.

        Returns:
            dict[str, typing.Any]: Control sequence configuration dict.
        """
        return {
            **asdict(self),
            "controls": [
                {**pulse.to_dict(), "_name": pulse.__class__.__name__}
                for pulse in self.controls
            ],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, typing.Any], controls: typing.Sequence[type[BaseControl]]
    ) -> "ControlSequence":
        """Construct the control sequence from dict.

        Args:
            data (dict[str, typing.Any]): Dict contain information for sequence construction
            control (typing.Sequence[type[BasePulse]]): Constructor of the controls

        Returns:
            ControlSequence: Instance of the control sequence.
        """
        parsed_data = []
        for d, pulse in zip(data["controls"], controls):
            assert isinstance(d, dict), f"Expected dict, got {type(d)}"

            # remove the _name key
            d.pop("_name")
            parsed_data.append(pulse.from_dict(d))

        data["controls"] = parsed_data
        data["validate"] = True

        return cls(**data)

    def to_file(self, path: typing.Union[str, pathlib.Path]):
        """Save configuration of the pulse to file given folder path.

        Args:
            path (typing.Union[str, pathlib.Path]): Path to the folder to save sequence, will be created if not existed.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)

        os.makedirs(path, exist_ok=True)
        with open(path / "control_sequence.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_file(
        cls,
        path: typing.Union[str, pathlib.Path],
        controls: typing.Sequence[type[BaseControl]],
    ) -> "ControlSequence":
        """Construct control seqence from path

        Args:
            path (typing.Union[str, pathlib.Path]): Path to configuration of control sequence.
            controls (typing.Sequence[type[BasePulse]]): Constructor of the control in the sequence.

        Returns:
            ControlSequence: Control sequence instance.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)

        with open(path / "control_sequence.json", "r") as f:
            dict_control_sequence = json.load(f)

        return cls.from_dict(dict_control_sequence, controls=controls)


def array_to_list_of_params(
    array: jnp.ndarray, parameter_structure: list[list[str]]
) -> list[ParametersDictType]:
    """Convert the array of control parameter to the list form

    Args:
        array (jnp.ndarray): Control parameter array
        parameter_structure (list[list[str]]): The structure of the control sequence

    Returns:
        list[ParametersDictType]: Control parameter in the list form.
    """
    temp: list[ParametersDictType] = []
    idx = 0
    for sub_pulse in parameter_structure:
        temp_dict: ParametersDictType = {}
        for param in sub_pulse:
            temp_dict[param] = array[idx]
            idx += 1
        temp.append(temp_dict)

    return temp


def list_of_params_to_array(
    params: list[ParametersDictType], parameter_structure: list[list[str]]
) -> jnp.ndarray:
    """Convert the control parameter in the list form to flatten array form

    Args:
        params (list[ParametersDictType]): Control parameter in the list form
        parameter_structure (list[list[str]]): The structure of the control sequence

    Returns:
        jnp.ndarray: Control parameters array
    """
    temp = []
    for subp_idx, sub_pulse in enumerate(parameter_structure):
        for param in sub_pulse:
            temp.append(params[subp_idx][param])

    return jnp.array(temp)


def get_param_array_converter(control_sequence: ControlSequence):
    """This function returns two functions that can convert between a list of parameter dictionaries and a flat array.

    >>> array_to_list_of_params_fn, list_of_params_to_array_fn = get_param_array_converter(control_sequence)

        Args:
        control_sequence (ControlSequence): The pulse sequence object.

    Returns:
        typing.Any: A tuple containing two functions. The first function converts an array to a list of parameter dictionaries, and the second function converts a list of parameter dictionaries to an array.
    """
    structure = control_sequence.get_parameter_names()

    def array_to_list_of_params_fn(
        array: jnp.ndarray,
    ) -> list[ParametersDictType]:
        return array_to_list_of_params(array, structure)

    def list_of_params_to_array_fn(
        params: list[ParametersDictType],
    ) -> jnp.ndarray:
        return list_of_params_to_array(params, structure)

    return array_to_list_of_params_fn, list_of_params_to_array_fn


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

        with open(path / "control_sequence.json", "r") as f:
            control_sequence_dict = json.load(f)

        parsed_controls = []

        for pulse_dict in control_sequence_dict["controls"]:
            for control_class in controls_list:
                if pulse_dict["_name"] == control_class.__name__:
                    parsed_controls.append(control_class)

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
    structure = control_sequence.get_parameter_names()

    def array_to_list_of_params_fn(array: jnp.ndarray):
        return array_to_list_of_params(array, structure)

    def get_envelope(params: jnp.ndarray) -> typing.Callable[..., typing.Any]:
        return control_sequence.get_envelope(array_to_list_of_params_fn(params))

    return get_envelope
