<!--
---
weight: 433
title: "Pyne Edge Profile"
description: "Look up the E codes of the Pyne Edge profile check: the Python constructs and types a script outside the profile uses, and how to stay inside it."
icon: "verified"
date: "2026-11-15"
lastmod: "2026-11-15"
draft: false
toc: true
categories: ["Reference"]
tags: ["error-codes", "pyne-edge"]
---
-->

# Pyne Edge Profile

<a id="E0001"></a>

## E0001: Syntax outside the Edge profile

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
{construct} is not allowed in the Pyne Edge profile.
```

The Pyne Edge profile is the subset of Python that is enough for everything a
Pine Script program needs: every Pine script can be written in it. This
construct is outside it; a full `@pyne` script can use it.

<a id="E0002"></a>

## E0002: Python built-in instead of the Pine one

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is Python's built-in here, which is not allowed in the Pyne Edge profile. For {meaning}, import it: `from pynecore.lib import {name}`.
```

Without the import, the name means Python's own built-in, not the Pine
function of the same name.

<a id="E0003"></a>

## E0003: lambda outside a field default

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`lambda` is allowed only as a `field(default_factory=...)` default of a field.
```

The Pyne Edge profile has no function values. The one `lambda` it takes
builds a field's default value.

<a id="E0004"></a>

## E0004: assert message that is not a string literal

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
An `assert` message must be a string literal in the Pyne Edge profile.
```

The message is fixed text, as Pine's own messages are: the profile has no use
for one built while the script runs.

<a id="E0005"></a>

## E0005: Assignment to a subscript

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
Assigning to a subscript is not allowed in the Pyne Edge profile; use `array.set()`.
```

An array element is set with `array.set(id, index, value)`.

<a id="E0006"></a>

## E0006: Import outside the Edge profile

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
{import} is not allowed in the Pyne Edge profile: only the PyneCore API and `lib.*` libraries can be imported.
```

Relative imports, star imports and modules other than the PyneCore API and
the user's `lib.*` libraries are outside the profile.

<a id="E0007"></a>

## E0007: Decorator outside the Edge profile

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
Only `@script.indicator/strategy/library(...)`, `@method` and `@export` can decorate a function in the Pyne Edge profile.
```

Other decorators change a function while the script runs, which nothing in
Pine needs.

<a id="E0008"></a>

## E0008: Class outside the Edge profile

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
{detail} A class in the Pyne Edge profile is a `@udt` or `@dataclass` field list with methods, or an enum.
```

A class has exactly one `@udt` or `@dataclass` decorator, no base classes or
keywords, and a body of annotated fields and undecorated methods that take
their receiver first.

An enum is the other class form: its single base is `StrEnum` imported from
`enum`, it takes no decorator, and its body holds `member = 'title'`
assignments only.

<a id="E0009"></a>

## E0009: Call outside the Edge profile

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` cannot be called in the Pyne Edge profile: only defined functions, imported names and {builtins} can.
```

A call goes to a function of the script, an imported name, or one of the few
Python built-ins the profile keeps.

<a id="E0010"></a>

## E0010: Attribute set on a function or module

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
Assigning an attribute on `{root}` is not allowed in the Pyne Edge profile: functions and modules are not objects.
```

Only objects of user-defined types have fields to set.

<a id="E0011"></a>

## E0011: Double-underscore name

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The name `{name}` is not allowed in the Pyne Edge profile: double-underscore names belong to PyneCore's transform.
```

PyneCore's transform names its own functions and variables with double
underscores, and a Pine script has no such names. Rename the identifier.

<a id="E0012"></a>

## E0012: Name without a known type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` has no known type.
```

The Pyne Edge profile types every value the way Pine does. The value this name
holds could not be typed; give it a value of a known type, or annotate it.

<a id="E0013"></a>

## E0013: Function used as a value

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is a function, not a value.
```

Pine has no function values: a function is called where it is used, never
stored or passed.

<a id="E0014"></a>

## E0014: Call without a known type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The call to `{callee}` has no known type.
```

What the called function returns could not be typed. Annotate its return, or
the parameters its result depends on.

<a id="E0015"></a>

## E0015: Call of an unknown callee

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
This call has no known type.
```

The called expression is not a name, so what it returns cannot be typed. Call
a defined function or a lib name.

<a id="E0016"></a>

## E0016: Method call without a known type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The call to the method `{method}` has no known type.
```

What the method returns could not be typed. Annotate its return, or its
receiver's type.

<a id="E0017"></a>

## E0017: Lib call without a known type here

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` has no known type here.
```

The lib function has no overload for the argument types it gets here. Give its
arguments known types.

<a id="E0018"></a>

## E0018: Unknown lib name

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is not a lib name.
```

The name is not part of `pynecore.lib`. Check its spelling.

<a id="E0019"></a>

## E0019: Field without a known type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`.{field}` has no known type on `{type}`.
```

The field could not be typed on this value. Declare the field on a `@udt`
class.

<a id="E0020"></a>

## E0020: Index without a known type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
Indexing `{type}` has no known type.
```

Only a series, a tuple, or an array read through `array.get()` can be indexed
with a known result type.

<a id="E0021"></a>

## E0021: Operation without a known type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{expression}` has no known type over {operands}.
```

The operation does not take these operand types together. Give the operands
types the operation accepts.

<a id="E0022"></a>

## E0022: Expression outside Pine

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{expression}` is not a Pine expression.
```

The expression has no Pine counterpart. Write it with Pine constructs only.

<a id="E0023"></a>

## E0023: Two types for one value

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
{what} gets both `{first}` and `{second}`, which are different types.
```

A Pine variable has one type. Make every branch, or every assignment, give it
the same type.

<a id="E0024"></a>

## E0024: Starred unpacking target

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
A starred target takes an unknown number of elements, which Pine has no form for.
```

Unpack exactly as many names as the tuple has elements.

<a id="E0025"></a>

## E0025: Unpacking count mismatch

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{type}` has {count} elements, and this unpacks {names}.
```

Unpack as many names as the tuple has elements.

<a id="E0026"></a>

## E0026: Overload differs per call context

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The overload chosen for this call differs per call context: {pins}.
```

The same call site would take a different overload depending on where its
function is called from. Annotate the parameters so every call site agrees.

<a id="E0027"></a>

## E0027: Conversion differs per call context

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The implicit conversion of a value differs per call context, and not every instance can be given its own.
```

Annotate the parameters so every call site passes the same types.

<a id="E0028"></a>

## E0028: Collection operand without a known type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{type}` takes an operand whose type is not known here, so what it holds afterwards is not known either.
```

Give the operand a known type.

<a id="E0029"></a>

## E0029: Collection operand of another element type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{type}` cannot take `{other}`: the elements are different types.
```

Pass a collection of the same element type.

<a id="E0030"></a>

## E0030: Function name assigned

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is assigned as well as defined, so what it calls is unknown.
```

A call is typed from the function its name defines. Call it under a name
nothing assigns to.

<a id="E0031"></a>

## E0031: Name imported twice

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is imported more than once, so what it calls is unknown.
```

Import the name once, under a name nothing else binds.

<a id="E0032"></a>

## E0032: Imported name assigned

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is assigned as well as imported, so what it calls is unknown.
```

Call the import under a name nothing assigns to.

<a id="E0033"></a>

## E0033: Re-entrant call

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}` is re-entrant, so its result type is unknown.
```

A recursive call is typed from the function's annotations. Annotate the
parameters and the return of the recursive function.

<a id="E0034"></a>

## E0034: Too many call contexts

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}` hit the per-module context limit.
```

The function is called with too many distinct argument types. Call it with
fewer, or annotate its parameters.

<a id="E0035"></a>

## E0035: Import without annotations

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is imported from `{source}`, where its parameters carry no annotations, so what it returns is unknown.
```

An imported function is typed from its annotations. Annotate its parameters in
the module it comes from.

<a id="E0036"></a>

## E0036: Import of a reported module

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{module}` is reported, and no type it publishes may drive a dispatch here: {reason}.
```

The imported module has errors of its own. Fix that module first.

<a id="E0037"></a>

## E0037: Import cycle

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{module}` imports this module back, so its signatures are not available yet.
```

Break the import cycle between the two modules.

<a id="E0038"></a>

## E0038: For loop over a non-iterable

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{expression}` is a `{type}`; a for loop iterates a range or an array.
```

Loop over `range(...)` or an array.

<a id="E0039"></a>

## E0039: Indexing outside Pine

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
Indexing `{type}` is not Pine.
```

Read elements with `array.get`, `matrix.get` or `map.get`.

<a id="E0040"></a>

## E0040: History of a plain value

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{expression}` is a plain `{type}`, not a series.
```

Only a series keeps its history. Declare the variable as a `Series` to read
its past values.

<a id="E0041"></a>

## E0041: Assignment of another type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is declared `{declared}` and assigned `{assigned}`.
```

Declare the variable with the type it is assigned, or assign the declared
type.

<a id="E0042"></a>

## E0042: Field assigned another type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{field}` holds `{declared}` and is assigned `{assigned}`.
```

Assign a value of the field's type.

<a id="E0043"></a>

## E0043: Unknown function

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is not a function this module defines or imports, nor a lib name.
```

Define the function, import it, or call a lib name.

<a id="E0044"></a>

## E0044: Callee is not a name

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The callee is not a name.
```

Call a defined function or a lib name.

<a id="E0045"></a>

## E0045: Field of a value of unknown class

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The class of `{value}` is not known here, so its field `{field}` has no type.
```

Annotate the value with the type it holds.

<a id="E0046"></a>

## E0046: Method of a value of unknown class

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The class of `{value}` is not known here, so its method `{method}` has no type.
```

Annotate the receiver with the type it holds.

<a id="E0047"></a>

## E0047: Field set on a value of unknown class

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The class of `{value}` is not known here, so its field `{field}` cannot be checked.
```

Annotate the value with the type it holds.

<a id="E0048"></a>

## E0048: Unknown field

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{owner}` has no field `{field}`.
```

Declare the field on the class, or use a field it has.

<a id="E0049"></a>

## E0049: Field set on a value without fields

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
A `{type}` has no field `{field}` to assign.
```

Only objects of a `@udt` class have fields to set.

<a id="E0050"></a>

## E0050: Unknown loop counter field

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The loop counter has no field `{field}`.
```

Read `value` or `entering`, or call `step` or `enter`.

<a id="E0051"></a>

## E0051: Lib namespace called

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is a lib namespace, not a function.
```

Call a function in the namespace.

<a id="E0052"></a>

## E0052: Lib value called

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{name}` is a lib value, not a function.
```

Read the value without calling it.

<a id="E0053"></a>

## E0053: Unknown method

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{method}` is not a method of `{type}` here.
```

Declare the method on the receiver's class, or call a method its type has.

<a id="E0054"></a>

## E0054: Method not reachable by name

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{method}` is declared on `{type}`, but no export the runtime searches by name holds it.
```

Pass the method itself: `method_call(method, receiver, ...)`.

<a id="E0055"></a>

## E0055: Loop bounds without a known type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The bounds of `{expression}` are not known here, so its counter has no type.
```

Give the loop bounds known numeric types.

<a id="E0056"></a>

## E0056: Bound is not a number

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}` takes a number for a bound, `{type}` passed.
```

Pass `int` or `float` bounds.

<a id="E0057"></a>

## E0057: step() arity

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`step` takes 1 argument, {count} passed.
```

Pass the bound: `step(to)`.

<a id="E0058"></a>

## E0058: enter() arity

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`enter` takes 0 or 1 argument(s), {count} passed.
```

Pass the step: `enter(by)`.

<a id="E0059"></a>

## E0059: method_call() arguments

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`method_call` takes a method and a receiver.
```

Spell it `method_call(method, receiver, ...)`.

<a id="E0060"></a>

## E0060: Wrong number of arguments

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}` does not take {count} argument(s).
```

Call the function the way it is declared.

<a id="E0061"></a>

## E0061: Too few arguments

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}` does not take {count} argument(s) (at least {least} needed).
```

Pass every argument the function needs.

<a id="E0062"></a>

## E0062: Wrong number of positional arguments

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}` takes {positional} positional argument(s), {count} passed.
```

Call the function the way it is declared.

<a id="E0063"></a>

## E0063: Unknown parameter

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}` has no parameter `{parameter}`.
```

Pass only the parameters the function declares.

<a id="E0064"></a>

## E0064: Argument of another type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}` takes `{declared}` for `{parameter}`, `{passed}` passed.
```

Pass a value of the declared type.

<a id="E0065"></a>

## E0065: No matching overload

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
No overload of `{function}` takes these arguments.
```

Call it with argument types one of its overloads declares.

<a id="E0066"></a>

## E0066: Argument passed twice

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{parameter}` is passed to `{function}` twice.
```

Pass each argument once.

<a id="E0067"></a>

## E0067: Missing argument

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{function}` needs an argument for `{parameter}`.
```

Pass every parameter that has no default.

<a id="E0068"></a>

## E0068: Wrong number of constructor arguments

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{type}` has {fields} field(s), {count} argument(s) passed.
```

Pass at most one argument per field.

<a id="E0069"></a>

## E0069: Missing field value

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{type}` needs a value for `{field}`.
```

Pass every field that has no default.

<a id="E0070"></a>

## E0070: Field value of another type

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
`{field}` holds `{declared}`, `{passed}` passed.
```

Pass a value of the field's type.

<a id="E0071"></a>

## E0071: Overload not decidable

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
Argument {argument} of `{function}` has no known type, so the overload this call runs is not decided.
```

An overloaded function is called with an argument whose type the analysis does
not know: an instance of a plain class instead of a `@udt` one, or the value of
a helper without a return annotation. Which overload runs is then decided from
the values at run time, and an argument no overload takes stops the run there.
Give the argument a known type: annotate it, or make its class a `@udt`.

<a id="E0072"></a>

## E0072: Overload not decidable

| Severity | PyneCore raises |
|----------|-----------------|
| error    | `SyntaxError`   |

Message:

```text
The argument types of `{function}` do not decide one overload for this call.
```

The static types of the arguments leave more than one overload of the function,
or none the analysis can read, for this call. Which overload runs is then
decided from the values at run time. Annotate the overloads' parameters, or pass
arguments whose types name one overload.
