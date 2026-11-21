<!--
---
weight: 430
title: "Diagnostic Codes"
description: "Every error code PyneCore reports, with its message and meaning: the run's R codes, the Pyne code rules (P) and the Pyne Edge profile (E)."
icon: "bug_report"
date: "2026-11-15"
lastmod: "2026-11-15"
draft: false
toc: true
categories: ["Reference"]
tags: ["error-codes", "debugging"]
---
-->

# Diagnostic Codes

Every code PyneCore reports. A run, the transform and the Pyne Edge profile
check each name their errors by code, so the code in a message leads here.

## [Run-Time Errors](runtime-errors.md)

| Code                             | Severity | Title                                         |
|----------------------------------|----------|-----------------------------------------------|
| [R0001](runtime-errors.md#R0001) | error    | Division by zero                              |
| [R0002](runtime-errors.md#R0002) | error    | An na value converted to an integer           |
| [R0003](runtime-errors.md#R0003) | error    | An infinite value converted to an integer     |
| [R0004](runtime-errors.md#R0004) | error    | Assertion failed                              |
| [R0005](runtime-errors.md#R0005) | error    | runtime.error()                               |
| [R0006](runtime-errors.md#R0006) | error    | Argument must be greater than 0               |
| [R0007](runtime-errors.md#R0007) | error    | Argument must not be negative                 |
| [R0008](runtime-errors.md#R0008) | error    | Argument must not be 0                        |
| [R0009](runtime-errors.md#R0009) | error    | Percentage out of range                       |
| [R0010](runtime-errors.md#R0010) | error    | Index out of range                            |
| [R0011](runtime-errors.md#R0011) | error    | Empty collection                              |
| [R0012](runtime-errors.md#R0012) | error    | Sizes do not match                            |
| [R0013](runtime-errors.md#R0013) | error    | Matrix is not square                          |
| [R0014](runtime-errors.md#R0014) | error    | Matrix is singular                            |
| [R0015](runtime-errors.md#R0015) | error    | Calculation did not converge                  |
| [R0016](runtime-errors.md#R0016) | error    | An na object                                  |
| [R0017](runtime-errors.md#R0017) | error    | Variable read before it is assigned           |
| [R0018](runtime-errors.md#R0018) | error    | Resource limit exceeded                       |
| [R0019](runtime-errors.md#R0019) | error    | Invalid timeframe                             |
| [R0020](runtime-errors.md#R0020) | error    | Tick timeframe                                |
| [R0021](runtime-errors.md#R0021) | error    | Invalid session                               |
| [R0022](runtime-errors.md#R0022) | error    | Invalid date                                  |
| [R0023](runtime-errors.md#R0023) | error    | Invalid format pattern                        |
| [R0024](runtime-errors.md#R0024) | error    | Invalid regular expression                    |
| [R0025](runtime-errors.md#R0025) | error    | Unknown time zone                             |
| [R0026](runtime-errors.md#R0026) | error    | No data for request.security                  |
| [R0027](runtime-errors.md#R0027) | error    | Lower timeframe request of a higher timeframe |
| [R0028](runtime-errors.md#R0028) | error    | request.security that varies in a loop        |
| [R0029](runtime-errors.md#R0029) | error    | Strategy value is not finite                  |
| [R0030](runtime-errors.md#R0030) | error    | Strategy read inside request.security         |
| [R0031](runtime-errors.md#R0031) | error    | line.get_price on a time-based line           |
| [R0032](runtime-errors.md#R0032) | error    | Invalid operand                               |
| [R0033](runtime-errors.md#R0033) | error    | Number out of range                           |
| [R0034](runtime-errors.md#R0034) | error    | Invalid decimal operation                     |
| [R0035](runtime-errors.md#R0035) | error    | Not available                                 |
| [R0036](runtime-errors.md#R0036) | error    | Script timeframe does not fit the data        |
| [R0037](runtime-errors.md#R0037) | error    | Invalid data                                  |
| [R0038](runtime-errors.md#R0038) | error    | Invalid drawing argument                      |
| [R0039](runtime-errors.md#R0039) | error    | No such method                                |
| [R0040](runtime-errors.md#R0040) | error    | Arguments in the wrong order                  |
| [R0041](runtime-errors.md#R0041) | error    | Matrix with na elements                       |
| [R0042](runtime-errors.md#R0042) | error    | Invalid run option                            |
| [R0043](runtime-errors.md#R0043) | error    | Field of an na object                         |
| [R0044](runtime-errors.md#R0044) | error    | Copy of an na object                          |
| [R0045](runtime-errors.md#R0045) | error    | Symbol detail not given                       |
| [R0046](runtime-errors.md#R0046) | error    | Unreadable data file                          |
| [R0047](runtime-errors.md#R0047) | error    | Script cannot run                             |
| [R0048](runtime-errors.md#R0048) | error    | File not found                                |
| [R0049](runtime-errors.md#R0049) | error    | Data provider unavailable                     |
| [R0050](runtime-errors.md#R0050) | error    | Invalid call arguments                        |
| [R0051](runtime-errors.md#R0051) | error    | Invalid series subscript                      |
| [R0052](runtime-errors.md#R0052) | error    | Argument out of range                         |
| [R0053](runtime-errors.md#R0053) | error    | Export used wrongly                           |
| [R0054](runtime-errors.md#R0054) | error    | request.security called outside the transform |
| [R0055](runtime-errors.md#R0055) | error    | Symbol of a live request.security unavailable |
| [I0004](runtime-errors.md#I0004) | error    | Internal error                                |

## [Pyne Code Rules](pyne-code-rules.md)

| Code                              | Severity | Title                                          |
|-----------------------------------|----------|------------------------------------------------|
| [P0001](pyne-code-rules.md#P0001) | error    | Identifier with PyneCore's reserved character  |
| [P0002](pyne-code-rules.md#P0002) | error    | Identifier reserved by the transform           |
| [P0003](pyne-code-rules.md#P0003) | error    | Script decorator not on main                   |
| [P0004](pyne-code-rules.md#P0004) | error    | Second script decorator                        |
| [P0005](pyne-code-rules.md#P0005) | error    | na_bool is not a literal                       |
| [P0006](pyne-code-rules.md#P0006) | error    | Series variable outside a function             |
| [P0007](pyne-code-rules.md#P0007) | error    | Persistent variable outside a function         |
| [P0008](pyne-code-rules.md#P0008) | error    | Stateful call at module level                  |
| [P0009](pyne-code-rules.md#P0009) | error    | Input outside main's parameters                |
| [P0010](pyne-code-rules.md#P0010) | error    | lib imported under an alias                    |
| [P0011](pyne-code-rules.md#P0011) | error    | Wildcard import of a module without __all__    |
| [P0012](pyne-code-rules.md#P0012) | error    | Unknown lib attribute                          |
| [P0013](pyne-code-rules.md#P0013) | error    | Module-level object modified inside a function |
| [P0014](pyne-code-rules.md#P0014) | error    | Plot called conditionally                      |
| [P0015](pyne-code-rules.md#P0015) | error    | Plot used as a value                           |
| [P0016](pyne-code-rules.md#P0016) | error    | Stateful method of the wrong kind              |
| [P0017](pyne-code-rules.md#P0017) | error    | Method receiver from an unpacked argument      |
| [P0018](pyne-code-rules.md#P0018) | error    | Export captures a library global               |
| [P0019](pyne-code-rules.md#P0019) | error    | Strategy state in a security expression        |
| [P0020](pyne-code-rules.md#P0020) | error    | Drawing in a security expression               |
| [P0021](pyne-code-rules.md#P0021) | error    | Too many requests in one function              |

## [Pyne Edge Profile](edge-profile.md)

| Code                           | Severity | Title                                       |
|--------------------------------|----------|---------------------------------------------|
| [E0001](edge-profile.md#E0001) | error    | Syntax outside the Edge profile             |
| [E0002](edge-profile.md#E0002) | error    | Python built-in instead of the Pine one     |
| [E0003](edge-profile.md#E0003) | error    | lambda outside a field default              |
| [E0004](edge-profile.md#E0004) | error    | assert message that is not a string literal |
| [E0005](edge-profile.md#E0005) | error    | Assignment to a subscript                   |
| [E0006](edge-profile.md#E0006) | error    | Import outside the Edge profile             |
| [E0007](edge-profile.md#E0007) | error    | Decorator outside the Edge profile          |
| [E0008](edge-profile.md#E0008) | error    | Class outside the Edge profile              |
| [E0009](edge-profile.md#E0009) | error    | Call outside the Edge profile               |
| [E0010](edge-profile.md#E0010) | error    | Attribute set on a function or module       |
| [E0011](edge-profile.md#E0011) | error    | Double-underscore name                      |
| [E0012](edge-profile.md#E0012) | error    | Name without a known type                   |
| [E0013](edge-profile.md#E0013) | error    | Function used as a value                    |
| [E0014](edge-profile.md#E0014) | error    | Call without a known type                   |
| [E0015](edge-profile.md#E0015) | error    | Call of an unknown callee                   |
| [E0016](edge-profile.md#E0016) | error    | Method call without a known type            |
| [E0017](edge-profile.md#E0017) | error    | Lib call without a known type here          |
| [E0018](edge-profile.md#E0018) | error    | Unknown lib name                            |
| [E0019](edge-profile.md#E0019) | error    | Field without a known type                  |
| [E0020](edge-profile.md#E0020) | error    | Index without a known type                  |
| [E0021](edge-profile.md#E0021) | error    | Operation without a known type              |
| [E0022](edge-profile.md#E0022) | error    | Expression outside Pine                     |
| [E0023](edge-profile.md#E0023) | error    | Two types for one value                     |
| [E0024](edge-profile.md#E0024) | error    | Starred unpacking target                    |
| [E0025](edge-profile.md#E0025) | error    | Unpacking count mismatch                    |
| [E0026](edge-profile.md#E0026) | error    | Overload differs per call context           |
| [E0027](edge-profile.md#E0027) | error    | Conversion differs per call context         |
| [E0028](edge-profile.md#E0028) | error    | Collection operand without a known type     |
| [E0029](edge-profile.md#E0029) | error    | Collection operand of another element type  |
| [E0030](edge-profile.md#E0030) | error    | Function name assigned                      |
| [E0031](edge-profile.md#E0031) | error    | Name imported twice                         |
| [E0032](edge-profile.md#E0032) | error    | Imported name assigned                      |
| [E0033](edge-profile.md#E0033) | error    | Re-entrant call                             |
| [E0034](edge-profile.md#E0034) | error    | Too many call contexts                      |
| [E0035](edge-profile.md#E0035) | error    | Import without annotations                  |
| [E0036](edge-profile.md#E0036) | error    | Import of a reported module                 |
| [E0037](edge-profile.md#E0037) | error    | Import cycle                                |
| [E0038](edge-profile.md#E0038) | error    | For loop over a non-iterable                |
| [E0039](edge-profile.md#E0039) | error    | Indexing outside Pine                       |
| [E0040](edge-profile.md#E0040) | error    | History of a plain value                    |
| [E0041](edge-profile.md#E0041) | error    | Assignment of another type                  |
| [E0042](edge-profile.md#E0042) | error    | Field assigned another type                 |
| [E0043](edge-profile.md#E0043) | error    | Unknown function                            |
| [E0044](edge-profile.md#E0044) | error    | Callee is not a name                        |
| [E0045](edge-profile.md#E0045) | error    | Field of a value of unknown class           |
| [E0046](edge-profile.md#E0046) | error    | Method of a value of unknown class          |
| [E0047](edge-profile.md#E0047) | error    | Field set on a value of unknown class       |
| [E0048](edge-profile.md#E0048) | error    | Unknown field                               |
| [E0049](edge-profile.md#E0049) | error    | Field set on a value without fields         |
| [E0050](edge-profile.md#E0050) | error    | Unknown loop counter field                  |
| [E0051](edge-profile.md#E0051) | error    | Lib namespace called                        |
| [E0052](edge-profile.md#E0052) | error    | Lib value called                            |
| [E0053](edge-profile.md#E0053) | error    | Unknown method                              |
| [E0054](edge-profile.md#E0054) | error    | Method not reachable by name                |
| [E0055](edge-profile.md#E0055) | error    | Loop bounds without a known type            |
| [E0056](edge-profile.md#E0056) | error    | Bound is not a number                       |
| [E0057](edge-profile.md#E0057) | error    | step() arity                                |
| [E0058](edge-profile.md#E0058) | error    | enter() arity                               |
| [E0059](edge-profile.md#E0059) | error    | method_call() arguments                     |
| [E0060](edge-profile.md#E0060) | error    | Wrong number of arguments                   |
| [E0061](edge-profile.md#E0061) | error    | Too few arguments                           |
| [E0062](edge-profile.md#E0062) | error    | Wrong number of positional arguments        |
| [E0063](edge-profile.md#E0063) | error    | Unknown parameter                           |
| [E0064](edge-profile.md#E0064) | error    | Argument of another type                    |
| [E0065](edge-profile.md#E0065) | error    | No matching overload                        |
| [E0066](edge-profile.md#E0066) | error    | Argument passed twice                       |
| [E0067](edge-profile.md#E0067) | error    | Missing argument                            |
| [E0068](edge-profile.md#E0068) | error    | Wrong number of constructor arguments       |
| [E0069](edge-profile.md#E0069) | error    | Missing field value                         |
| [E0070](edge-profile.md#E0070) | error    | Field value of another type                 |
| [E0071](edge-profile.md#E0071) | error    | Overload not decidable                      |
| [E0072](edge-profile.md#E0072) | error    | Overload not decidable                      |
