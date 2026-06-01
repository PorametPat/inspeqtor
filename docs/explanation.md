# Design Decisions

## Why `jax`?

We chose [JAX](https://docs.jax.dev/en/latest/index.html) as the numerical framework for `inspeqtor` because of its automatic differentiation, pseudo-random number generation, just-in-time compilation capabilities, and functional-style interface. These features make it accessible to both the scientific community and the engineering teams who maintain production infrastructure. With JAX, we can easily deploy `inspeqtor` to CPU, GPU, and TPU machines. Furthermore, the JAX ecosystem provides a wide range of libraries and tools that can be easily integrated with `inspeqtor`, such as [Flax](https://flax.readthedocs.io/en/stable/) for neural networks, [Optax](https://optax.readthedocs.io/en/latest/) for optimization, [NumPyro](https://num.pyro.ai/en/stable/) for probabilistic programming, [Diffrax](https://docs.kidger.site/diffrax/) for differential equations, and many more. This allows us to leverage existing tools in the JAX ecosystem to enhance the functionality of `inspeqtor` and make it more versatile.

Currently, we perform just-in-time (JIT) compilation on functions that are commonly used in the inner loops of our algorithms. We may consider removing JIT compilation in the future if appropriate, as it can introduce overhead during compilation.

## Random number generation

Since we are a library that provides tools for the characterization and calibration of quantum devices, we need to generate random numbers for various purposes. With JAX, we can use the `jax.random` module to generate random numbers. However, JAX's random number generation follows a functional style, which means we need to pass a random key to the functions that generate random numbers. Once used, a random key is consumed and should not be reused. To generate new random numbers, we split the key to create a new one. This approach ensures that random numbers can be generated reproducibly. For example, to generate a random number, we can do the following:

```python
import jax

key = jax.random.key(0)  # Create a random key
key, subkey = jax.random.split(key)  # Split the key to get a new subkey
random_number = jax.random.normal(subkey)  # Generate a random number
```

For more details on JAX's random number generation, refer to the [JAX documentation](https://docs.jax.dev/en/latest/random-numbers.html).

## The `stable` module

If you look within the `inspeqtor`'s modules, you will see that there are `v1` and `v2` modules, and the `stable` module. The `v1` module contains the first version of the code, which is the one that I have been developing and using for a while. The `v2` module contains the newer implementation. The `stable` module imports from both `v1` and `v2`, and re-exports the functions and classes that are considered stable. This way, users can import from `stable` without worrying about the versioning, and I can still work on the `v2` module without breaking the existing code in `v1`.

Importing from `stable` is simple as follows:

```python
import inspeqtor as sq
```

You don't need to specify the version.
