# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import sympy

from braket.parametric.free_parameter_expression import FreeParameterExpression


def _wrap(sympy_fn, arg, *extra_args):
    inner = arg.expression if isinstance(arg, FreeParameterExpression) else arg
    return FreeParameterExpression(sympy_fn(inner, *extra_args))


def sin(x):
    return _wrap(sympy.sin, x)


def cos(x):
    return _wrap(sympy.cos, x)


def tan(x):
    return _wrap(sympy.tan, x)


def arcsin(x):
    return _wrap(sympy.asin, x)


def arccos(x):
    return _wrap(sympy.acos, x)


def arctan(x):
    return _wrap(sympy.atan, x)


def exp(x):
    return _wrap(sympy.exp, x)


def log(x):
    return _wrap(sympy.log, x)


def sqrt(x):
    return _wrap(sympy.sqrt, x)


def mod(x, m):
    return _wrap(sympy.Mod, x, m)


def ceiling(x):
    return _wrap(sympy.ceiling, x)


def floor(x):
    return _wrap(sympy.floor, x)
