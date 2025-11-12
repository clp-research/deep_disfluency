"""Package initializer for deep_disfluency.

Compatibility shim: provide Python-2-style raw_input in Python 3.

Some modules in this repo still call input(...) (a Python 2 built-in).
Under Python 3 raw_input was removed; `input(...)` has the same behavior.
This small shim defines `raw_input` as an alias to `input` when running
under Python 3 so legacy calls continue to work.
"""

try:
	# If running under Python 2, raw_input already exists; do nothing.
	raw_input  # noqa: F401
except NameError:
	# Under Python 3, make raw_input behave like input()
	raw_input = input
