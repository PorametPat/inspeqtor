# API Reference

## Stable API (Recommended)

The **stable** API is the recommended interface for production use. Import it directly from the main `inspeqtor` namespace:

```python
import inspeqtor as sq

# Access stable modules
sq.data
sq.control
sq.models
sq.optimize
sq.physics
sq.utils
sq.boed
```

The stable API provides a consistent interface that combines the best features from both `v2` and `experimental` modules. This is the namespace that will remain backwards-compatible going forward.

## Development APIs

### v1 API

The `v1` API is the original implementation and is fully functional. While it continues to work, we recommend using the stable API for new projects:

```python
import inspeqtor.v1 as sq1
```

### V2 API

The `v2` API represents the next-generation architecture. It is currently being stabilized and incorporated into the main stable API:

```python
import inspeqtor.v2 as sq2
```

!!! tip "Migration Path"
    The stable API is designed to make migration seamless. As `v2` matures, its components are exposed through the stable namespace, allowing you to adopt new features without changing your import statements.

## API Versioning Strategy

Our versioning approach prioritizes stability:

1. **Stable** (`inspeqtor.*`): Production-ready, backwards-compatible API
2. **V2** (`inspeqtor.v2.*`): New features being stabilized
3. **v1** (`inspeqtor.v1.*`): Legacy implementation (fully functional but not recommended for new code)
