# Design Decisions

## The `stable` module

If you look within the `inspeqtor`'s modules, you will see that there are `v1` and `v2` modules, and the `stable` module. The `v1` module contains the first version of the code, which is the one that I have been developing and using for a while. The `v2` module contains the newer implementation. The `stable` module imports from both `v1` and `v2`, and re-exports the functions and classes that are considered stable. This way, users can import from `stable` without worrying about the versioning, and I can still work on the `v2` module without breaking the existing code in `v1`.

Importing from `stable` is simple as follows:

```python
import inspeqtor as sq
```

You don't need to specify the version.
