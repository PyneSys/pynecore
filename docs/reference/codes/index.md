# Diagnostic codes

Every code PyneCore reports.

| Code              | Severity | Title                                          |
|-------------------|----------|------------------------------------------------|
| [E0001](E0001.md) | error    | Syntax outside the Edge profile                |
| [E0002](E0002.md) | error    | Python built-in instead of the Pine one        |
| [E0003](E0003.md) | error    | lambda outside a field default                 |
| [E0004](E0004.md) | error    | assert message that is not a string literal    |
| [E0005](E0005.md) | error    | Assignment to a subscript                      |
| [E0006](E0006.md) | error    | Import outside the Edge profile                |
| [E0007](E0007.md) | error    | Decorator outside the Edge profile             |
| [E0008](E0008.md) | error    | Class outside the Edge profile                 |
| [E0009](E0009.md) | error    | Call outside the Edge profile                  |
| [E0010](E0010.md) | error    | Attribute set on a function or module          |
| [E0011](E0011.md) | error    | Double-underscore name                         |
| [E0012](E0012.md) | error    | Name without a known type                      |
| [E0013](E0013.md) | error    | Function used as a value                       |
| [E0014](E0014.md) | error    | Call without a known type                      |
| [E0015](E0015.md) | error    | Call of an unknown callee                      |
| [E0016](E0016.md) | error    | Method call without a known type               |
| [E0017](E0017.md) | error    | Lib call without a known type here             |
| [E0018](E0018.md) | error    | Unknown lib name                               |
| [E0019](E0019.md) | error    | Field without a known type                     |
| [E0020](E0020.md) | error    | Index without a known type                     |
| [E0021](E0021.md) | error    | Operation without a known type                 |
| [E0022](E0022.md) | error    | Expression outside Pine                        |
| [E0023](E0023.md) | error    | Two types for one value                        |
| [E0024](E0024.md) | error    | Starred unpacking target                       |
| [E0025](E0025.md) | error    | Unpacking count mismatch                       |
| [E0026](E0026.md) | error    | Overload differs per call context              |
| [E0027](E0027.md) | error    | Conversion differs per call context            |
| [E0028](E0028.md) | error    | Collection operand without a known type        |
| [E0029](E0029.md) | error    | Collection operand of another element type     |
| [E0030](E0030.md) | error    | Function name assigned                         |
| [E0031](E0031.md) | error    | Name imported twice                            |
| [E0032](E0032.md) | error    | Imported name assigned                         |
| [E0033](E0033.md) | error    | Re-entrant call                                |
| [E0034](E0034.md) | error    | Too many call contexts                         |
| [E0035](E0035.md) | error    | Import without annotations                     |
| [E0036](E0036.md) | error    | Import of a reported module                    |
| [E0037](E0037.md) | error    | Import cycle                                   |
| [E0038](E0038.md) | error    | For loop over a non-iterable                   |
| [E0039](E0039.md) | error    | Indexing outside Pine                          |
| [E0040](E0040.md) | error    | History of a plain value                       |
| [E0041](E0041.md) | error    | Assignment of another type                     |
| [E0042](E0042.md) | error    | Field assigned another type                    |
| [E0043](E0043.md) | error    | Unknown function                               |
| [E0044](E0044.md) | error    | Callee is not a name                           |
| [E0045](E0045.md) | error    | Field of a value of unknown class              |
| [E0046](E0046.md) | error    | Method of a value of unknown class             |
| [E0047](E0047.md) | error    | Field set on a value of unknown class          |
| [E0048](E0048.md) | error    | Unknown field                                  |
| [E0049](E0049.md) | error    | Field set on a value without fields            |
| [E0050](E0050.md) | error    | Unknown loop counter field                     |
| [E0051](E0051.md) | error    | Lib namespace called                           |
| [E0052](E0052.md) | error    | Lib value called                               |
| [E0053](E0053.md) | error    | Unknown method                                 |
| [E0054](E0054.md) | error    | Method not reachable by name                   |
| [E0055](E0055.md) | error    | Loop bounds without a known type               |
| [E0056](E0056.md) | error    | Bound is not a number                          |
| [E0057](E0057.md) | error    | step() arity                                   |
| [E0058](E0058.md) | error    | enter() arity                                  |
| [E0059](E0059.md) | error    | method_call() arguments                        |
| [E0060](E0060.md) | error    | Wrong number of arguments                      |
| [E0061](E0061.md) | error    | Too few arguments                              |
| [E0062](E0062.md) | error    | Wrong number of positional arguments           |
| [E0063](E0063.md) | error    | Unknown parameter                              |
| [E0064](E0064.md) | error    | Argument of another type                       |
| [E0065](E0065.md) | error    | No matching overload                           |
| [E0066](E0066.md) | error    | Argument passed twice                          |
| [E0067](E0067.md) | error    | Missing argument                               |
| [E0068](E0068.md) | error    | Wrong number of constructor arguments          |
| [E0069](E0069.md) | error    | Missing field value                            |
| [E0070](E0070.md) | error    | Field value of another type                    |
| [I0004](I0004.md) | error    | Internal error                                 |
| [P0001](P0001.md) | error    | Identifier with PyneCore's reserved character  |
| [P0002](P0002.md) | error    | Identifier reserved by the transform           |
| [P0003](P0003.md) | error    | Script decorator not on main                   |
| [P0004](P0004.md) | error    | Second script decorator                        |
| [P0005](P0005.md) | error    | na_bool is not a literal                       |
| [P0006](P0006.md) | error    | Series variable outside a function             |
| [P0007](P0007.md) | error    | Persistent variable outside a function         |
| [P0008](P0008.md) | error    | Stateful call at module level                  |
| [P0009](P0009.md) | error    | Input outside main's parameters                |
| [P0010](P0010.md) | error    | lib imported under an alias                    |
| [P0011](P0011.md) | error    | Wildcard import of a module without __all__    |
| [P0012](P0012.md) | error    | Unknown lib attribute                          |
| [P0013](P0013.md) | error    | Module-level object modified inside a function |
| [P0014](P0014.md) | error    | Plot called conditionally                      |
| [P0015](P0015.md) | error    | Plot used as a value                           |
| [P0016](P0016.md) | error    | Stateful method of the wrong kind              |
| [P0017](P0017.md) | error    | Method receiver from an unpacked argument      |
| [P0018](P0018.md) | error    | Export captures a library global               |
| [P0019](P0019.md) | error    | Strategy state in a security expression        |
| [P0020](P0020.md) | error    | Drawing in a security expression               |
| [P0021](P0021.md) | error    | Too many requests in one function              |
| [R0001](R0001.md) | error    | Division by zero                               |
| [R0002](R0002.md) | error    | An na value converted to an integer            |
| [R0003](R0003.md) | error    | An infinite value converted to an integer      |
| [R0004](R0004.md) | error    | Assertion failed                               |
| [R0005](R0005.md) | error    | runtime.error()                                |
| [R0006](R0006.md) | error    | Argument must be greater than 0                |
| [R0007](R0007.md) | error    | Argument must not be negative                  |
| [R0008](R0008.md) | error    | Argument must not be 0                         |
| [R0009](R0009.md) | error    | Percentage out of range                        |
| [R0010](R0010.md) | error    | Index out of range                             |
| [R0011](R0011.md) | error    | Empty collection                               |
| [R0012](R0012.md) | error    | Sizes do not match                             |
| [R0013](R0013.md) | error    | Matrix is not square                           |
| [R0014](R0014.md) | error    | Matrix is singular                             |
| [R0015](R0015.md) | error    | Calculation did not converge                   |
| [R0016](R0016.md) | error    | An na object                                   |
| [R0017](R0017.md) | error    | Variable read before it is assigned            |
| [R0018](R0018.md) | error    | Resource limit exceeded                        |
| [R0019](R0019.md) | error    | Invalid timeframe                              |
| [R0020](R0020.md) | error    | Tick timeframe                                 |
| [R0021](R0021.md) | error    | Invalid session                                |
| [R0022](R0022.md) | error    | Invalid date                                   |
| [R0023](R0023.md) | error    | Invalid format pattern                         |
| [R0024](R0024.md) | error    | Invalid regular expression                     |
| [R0025](R0025.md) | error    | Unknown time zone                              |
| [R0026](R0026.md) | error    | No data for request.security                   |
| [R0027](R0027.md) | error    | Lower timeframe request of a higher timeframe  |
| [R0028](R0028.md) | error    | request.security that varies in a loop         |
| [R0029](R0029.md) | error    | Strategy value is not finite                   |
| [R0030](R0030.md) | error    | Strategy read inside request.security          |
| [R0031](R0031.md) | error    | line.get_price on a time-based line            |
| [R0032](R0032.md) | error    | Invalid operand                                |
| [R0033](R0033.md) | error    | Number out of range                            |
| [R0034](R0034.md) | error    | Invalid decimal operation                      |
| [R0035](R0035.md) | error    | Not available                                  |
| [R0036](R0036.md) | error    | Script timeframe does not fit the data         |
| [R0037](R0037.md) | error    | Invalid data                                   |
| [R0038](R0038.md) | error    | Invalid drawing argument                       |
| [R0039](R0039.md) | error    | No such method                                 |
| [R0040](R0040.md) | error    | Arguments in the wrong order                   |
| [R0041](R0041.md) | error    | Matrix with na elements                        |
| [R0042](R0042.md) | error    | Invalid run option                             |
| [R0043](R0043.md) | error    | Field of an na object                          |
| [R0044](R0044.md) | error    | Copy of an na object                           |
| [R0045](R0045.md) | error    | Symbol detail not given                        |
| [R0046](R0046.md) | error    | Unreadable data file                           |
| [R0047](R0047.md) | error    | Script cannot run                              |
| [R0048](R0048.md) | error    | File not found                                 |
| [R0049](R0049.md) | error    | Data provider unavailable                      |
