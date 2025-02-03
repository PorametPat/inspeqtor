import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import inspeqtor.experimental as sq
    from inspeqtor.legacy import visualization as vis

    from functools import partial

    return jax, jnp, mo, partial, sq, vis


@app.cell
def _(jnp, partial, sq):
    qubit_info = sq.predefined.get_mock_qubit_information()
    pulse_sequence = sq.predefined.get_gaussian_pulse_sequence(qubit_info)
    dt = 2 / 9

    t_eval = jnp.linspace(
        0,
        pulse_sequence.pulse_length_dt * dt,
        pulse_sequence.pulse_length_dt,
    )

    array_to_list_of_params_fn, list_of_params_to_array_fn = (
        sq.pulse.get_param_array_converter(pulse_sequence)
    )

    hamiltonian = partial(
        sq.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v5(
            get_envelope=sq.predefined.get_envelope_transformer(
                pulse_sequence=pulse_sequence
            ),
            drive_frequency=qubit_info.frequency,
            dt=dt,
        ),
    )

    hamiltonian = sq.utils.detune_hamiltonian(
        hamiltonian, 0.0005 * qubit_info.frequency
    )

    whitebox = partial(
        sq.physics.solver,
        t_eval=t_eval,
        hamiltonian=hamiltonian,
        y0=jnp.eye(2, dtype=jnp.complex64),
        t0=0,
        t1=pulse_sequence.pulse_length_dt * dt,
    )
    return (
        array_to_list_of_params_fn,
        dt,
        hamiltonian,
        list_of_params_to_array_fn,
        pulse_sequence,
        qubit_info,
        t_eval,
        whitebox,
    )


@app.cell
def _(jnp, list_of_params_to_array_fn, sq, whitebox):
    params: list[sq.typing.ParametersDictType] = [{"theta": jnp.array(jnp.pi / 2)}]  # type: ignore
    unitary = whitebox(list_of_params_to_array_fn(params))
    return params, unitary


@app.cell
def _(sq, unitary, vis):
    vis.plot_expvals_v2(sq.utils.calculate_expectation_values(unitary).T)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
