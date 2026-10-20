# ir.requirements

Dependency check for `ir`: name what is missing and how to install it.

`ir` imports its dependencies deep inside `ir/base.py` and friends; on an
interpreter that lacks them (a fresh server’s system `python3`) that surfaces
as a bare `ModuleNotFoundError: No module named 'numpy'` that says nothing
about `ir` or the fix. [`check_requirements()`](#ir.requirements.check_requirements) runs first, from
`ir/__init__.py`, using only the standard library, and raises
[`MissingRequirements`](#ir.requirements.MissingRequirements) (a [`ModuleNotFoundError`](https://docs.python.org/3/builtins/exceptions.html#ModuleNotFoundError)) listing every
missing import, the interpreter it was looked up in, and the install commands.
It also checks the version bounds `ir` declares for them (`cw>=0.1.1,<0.2`),
read from `ir`’s installed metadata, and raises [`UnsatisfiedRequirements`](#ir.requirements.UnsatisfiedRequirements)
for an installed dependency outside them, instead of an opaque error later.

### Module Attributes

| [`DIST`](#ir.requirements.DIST)     | the distribution whose declared requirements hold the version bounds   |
|-----------------------------------------------------------|------------------------------------------------------------------------|
| [`REQUIRED`](#ir.requirements.REQUIRED) | import name -> distribution name, for everything `import ir` needs.    |

### Functions

| [`check_requirements`](#ir.requirements.check_requirements)([required, bounds])       | Raise [`MissingRequirements`](#ir.requirements.MissingRequirements) unless every dependency is importable, then [`UnsatisfiedRequirements`](#ir.requirements.UnsatisfiedRequirements) unless each is within the bounds `ir` declares (*bounds*: `{name: specifier}`, default read from `ir`'s metadata).   |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`declared_bounds`](#ir.requirements.declared_bounds)([dist])                      | `{normalized dependency name: specifier}` that *dist*'s installed metadata declares for this environment (extras excluded; the specifiers of repeated names are joined); empty when *dist* is not installed or has none.                                                                                             |
| [`missing_requirements`](#ir.requirements.missing_requirements)([required])             | Distribution names in *required* whose import name cannot be found.                                                                                                                                                                                                                                                  |
| [`requirements_message`](#ir.requirements.requirements_message)(missing)                | The actionable message for *missing* distributions.                                                                                                                                                                                                                                                                  |
| [`unsatisfied_message`](#ir.requirements.unsatisfied_message)(unsatisfied)             | The actionable message for *unsatisfied* `(dist, installed, specifier)`.                                                                                                                                                                                                                                             |
| [`unsatisfied_requirements`](#ir.requirements.unsatisfied_requirements)([required, bounds]) | `(distribution, installed version, specifier)` of each dependency in *required* that is installed outside its declared bounds.                                                                                                                                                                                       |
| [`version_satisfies`](#ir.requirements.version_satisfies)(installed, specifier)      | Whether *installed* meets *specifier* (`'>=0.1.1,<0.2'`).                                                                                                                                                                                                                                                            |

### Exceptions

| [`MissingRequirements`](#ir.requirements.MissingRequirements)     | `ir`'s dependencies are missing from the running interpreter.            |
|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`UnsatisfiedRequirements`](#ir.requirements.UnsatisfiedRequirements) | A dependency is installed in a version outside the bounds `ir` declares. |

### ir.requirements.DIST *= 'ir'*

the distribution whose declared requirements hold the version bounds

### *exception* ir.requirements.MissingRequirements

Bases: [`ModuleNotFoundError`](https://docs.python.org/3/builtins/exceptions.html#ModuleNotFoundError)

`ir`’s dependencies are missing from the running interpreter.

### ir.requirements.REQUIRED *: [Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [str](https://docs.python.org/3/builtins/stdtypes.html#str)]* *= {'cw': 'cw', 'dol': 'dol', 'ef': 'ef', 'numpy': 'numpy', 'vd': 'vd'}*

import name -> distribution name, for everything `import ir` needs.

### *exception* ir.requirements.UnsatisfiedRequirements

Bases: [`ImportError`](https://docs.python.org/3/builtins/exceptions.html#ImportError)

A dependency is installed in a version outside the bounds `ir` declares.

### ir.requirements.check_requirements(required={'cw': 'cw', 'dol': 'dol', 'ef': 'ef', 'numpy': 'numpy', 'vd': 'vd'}, , bounds=None)

Raise [`MissingRequirements`](#ir.requirements.MissingRequirements) unless every dependency is importable, then
[`UnsatisfiedRequirements`](#ir.requirements.UnsatisfiedRequirements) unless each is within the bounds `ir` declares
(*bounds*: `{name: specifier}`, default read from `ir`’s metadata).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### ir.requirements.declared_bounds(dist='ir')

`{normalized dependency name: specifier}` that *dist*’s installed metadata
declares for this environment (extras excluded; the specifiers of repeated names
are joined); empty when *dist* is not installed or has none.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.requirements.missing_requirements(required={'cw': 'cw', 'dol': 'dol', 'ef': 'ef', 'numpy': 'numpy', 'vd': 'vd'})

Distribution names in *required* whose import name cannot be found.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### ir.requirements.requirements_message(missing)

The actionable message for *missing* distributions.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.requirements.unsatisfied_message(unsatisfied)

The actionable message for *unsatisfied* `(dist, installed, specifier)`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### ir.requirements.unsatisfied_requirements(required={'cw': 'cw', 'dol': 'dol', 'ef': 'ef', 'numpy': 'numpy', 'vd': 'vd'}, , bounds=None)

`(distribution, installed version, specifier)` of each dependency in
*required* that is installed outside its declared bounds.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]

### ir.requirements.version_satisfies(installed, specifier)

Whether *installed* meets *specifier* (`'>=0.1.1,<0.2'`).

Uses `packaging` when it is installed. Without it, plain releases (`1.2.3`)
are compared with the standard library, and anything it cannot judge (a
pre-release, `dev`, `post` or local version, an unreadable specifier) is taken
to hold: this check must never block `import ir` on a doubt.

```pycon
>>> version_satisfies("0.1.6", ">=0.1.1,<0.2"), version_satisfies("0.2.0", "<0.2")
(True, False)
>>> version_satisfies("1.4", "~=1.2"), version_satisfies("2.0", "~=1.2")
(True, False)
```

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
