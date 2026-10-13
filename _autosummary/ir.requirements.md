# ir.requirements

Dependency check for `ir`: name what is missing and how to install it.

`ir` imports its dependencies deep inside `ir/base.py` and friends; on an
interpreter that lacks them (a fresh server’s system `python3`) that surfaces
as a bare `ModuleNotFoundError: No module named 'numpy'` that says nothing
about `ir` or the fix. [`check_requirements()`](#ir.requirements.check_requirements) runs first, from
`ir/__init__.py`, using only the standard library, and raises
[`MissingRequirements`](#ir.requirements.MissingRequirements) (a [`ModuleNotFoundError`](https://docs.python.org/3/builtins/exceptions.html#ModuleNotFoundError)) listing every
missing import, the interpreter it was looked up in, and the install commands.

### Module Attributes

| [`REQUIRED`](#ir.requirements.REQUIRED)   | import name -> distribution name, for everything `import ir` needs.   |
|-------------------------------------------------------------|-----------------------------------------------------------------------|

### Functions

| [`check_requirements`](#ir.requirements.check_requirements)([required])   | Raise [`MissingRequirements`](#ir.requirements.MissingRequirements) unless every dependency is importable.   |
|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`missing_requirements`](#ir.requirements.missing_requirements)([required]) | Distribution names in *required* whose import name cannot be found.                                                 |
| [`requirements_message`](#ir.requirements.requirements_message)(missing)    | The actionable message for *missing* distributions.                                                                 |

### Exceptions

| [`MissingRequirements`](#ir.requirements.MissingRequirements)   | `ir`'s dependencies are missing from the running interpreter.   |
|------------------------------------------------------------------------|-----------------------------------------------------------------|

### *exception* ir.requirements.MissingRequirements

Bases: [`ModuleNotFoundError`](https://docs.python.org/3/builtins/exceptions.html#ModuleNotFoundError)

`ir`’s dependencies are missing from the running interpreter.

### ir.requirements.REQUIRED *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [str](https://docs.python.org/3/builtins/stdtypes.html#str)]* *= {'cw': 'cw', 'dol': 'dol', 'ef': 'ef', 'numpy': 'numpy', 'vd': 'vd'}*

import name -> distribution name, for everything `import ir` needs.

### ir.requirements.check_requirements(required={'cw': 'cw', 'dol': 'dol', 'ef': 'ef', 'numpy': 'numpy', 'vd': 'vd'})

Raise [`MissingRequirements`](#ir.requirements.MissingRequirements) unless every dependency is importable.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.requirements.missing_requirements(required={'cw': 'cw', 'dol': 'dol', 'ef': 'ef', 'numpy': 'numpy', 'vd': 'vd'})

Distribution names in *required* whose import name cannot be found.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.requirements.requirements_message(missing)

The actionable message for *missing* distributions.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
