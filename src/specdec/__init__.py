"""specdec: single source of truth for long-context speculative decoding work.

Experiments live in specdec.experiments and import from the package; nothing
in the package imports from experiments. The selector repo is reached only
through specdec.models.selector.
"""
