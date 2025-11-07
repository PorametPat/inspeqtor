# Design Decisions

## The `import` 🧐

Currently, we recommend importing the package with the following code:

```python

import inspeqtor.experimental as sq

```

In the future, once the `legacy` module has been fully removed, the `experimental` submodule will likely be eliminated as well. Migrating should then be as simple as omitting `experimental` from your import statement.

We are now developing a `v2` API. Although the name could be misleading, this is our current approach. The `v2` API will gradually take the place of the `experimental` API. Since both APIs use the same namespace, migration should be straightforward.

For users wanting to try the new `v2` API, we recommend the following imports:

```python

import inspeqtor.experimental as sqe

import inspeqtor as sq

```

This demonstrates that `v2` provides a stable way to interact with `inspeqtor`, eliminating the need to specifically import `inspeqtor.v2 as sq` as with the `experimental` API.
