
# Design Decisions

## The `import` 🧐

We currently recommend user to import the package using the following code snippet.

``` { .python }
import inspeqtor.experimental as sq
```

In the future, the `experimental` is likely to be removed once the `legacy` module is fully removed. We expect the migration to be as easy as removing `experimental` part.