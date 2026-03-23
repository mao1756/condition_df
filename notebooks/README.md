# Notebook organization

Shared notebook UI helpers now live in `notebooks/support.py`.

That module centralizes the repeated Plotly trace builders, animation wiring,
and renderer setup that used to be copied across many notebooks.  The notebooks
are intentionally left as thin experiment drivers so the mathematical setup for
each example is still easy to follow.

The `junk/` directory is kept as an archive of exploratory experiments, but it
now reuses the same shared helper module instead of carrying its own copies of
animation boilerplate.
