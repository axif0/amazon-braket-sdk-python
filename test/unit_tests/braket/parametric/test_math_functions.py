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

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Number
from typing import Any

import pytest
import sympy

from braket.circuits import Circuit
from braket.circuits.serialization import IRType
from braket.devices import LocalSimulator
from braket.parametric import (
    FreeParameter,
    FreeParameterExpression,
    arccos,
    arcsin,
    arctan,
    ceiling,
    cos,
    exp,
    floor,
    log,
    mod,
    sin,
    sqrt,
    tan,
)

MathFunction = Callable[..., FreeParameterExpression]

OPENQASM_BUILTIN_FUNCTIONS: list[tuple[MathFunction, str]] = [
    (sin, "sin"),
    (cos, "cos"),
    (tan, "tan"),
    (arcsin, "arcsin"),
    (arccos, "arccos"),
    (arctan, "arctan"),
    (exp, "exp"),
    (log, "log"),
    (sqrt, "sqrt"),
    (ceiling, "ceiling"),
    (floor, "floor"),
]

# Sympy function types whose default printer names differ from OpenQASM 3.
SYMPY_TO_OPENQASM_PRINTER_CASES: list[tuple[Any, str]] = [
    (sympy.asin, "arcsin"),
    (sympy.acos, "arccos"),
    (sympy.atan, "arctan"),
    (sympy.Mod, "mod"),
]

STRING_FUNCTION_EXPRESSIONS: list[tuple[str, str]] = [
    ("sin(alpha)", "sin(alpha)"),
    ("cos(theta/2)", "cos(theta/2)"),
    ("arcsin(alpha)", "arcsin(alpha)"),
    ("arccos(alpha)", "arccos(alpha)"),
    ("arctan(alpha)", "arctan(alpha)"),
    ("mod(x, 2)", "mod(x, 2)"),
    ("exp(alpha)", "exp(alpha)"),
    ("sqrt(alpha)", "sqrt(alpha)"),
]


@pytest.fixture
def theta() -> FreeParameter:
    return FreeParameter("theta")


@pytest.fixture
def alpha() -> FreeParameter:
    return FreeParameter("alpha")


@pytest.mark.parametrize(("func", "name"), OPENQASM_BUILTIN_FUNCTIONS)
def test_math_function_helpers(
    func: MathFunction,
    name: str,
    theta: FreeParameter,
    alpha: FreeParameter,
) -> None:
    # --- constructor ---
    expr = func(theta)
    assert isinstance(expr, FreeParameterExpression)

    # --- OpenQASM string / repr ---
    assert str(expr) == f"{name}(theta)"
    assert repr(expr) == f"{name}(theta)"

    # --- partial subs ---
    # sqrt is skipped: sympy distributes sqrt(2*theta) → sqrt(2)*sqrt(theta),
    # so the string representation changes but the expression is still valid.
    if name != "sqrt":
        expr2 = func(theta * alpha)
        subbed_partial = expr2.subs({"alpha": 2})
        assert str(subbed_partial) == f"{name}(2*theta)"

    # --- full subs → numeric result ---
    subbed_full = expr.subs({"theta": 0.5})
    assert isinstance(subbed_full, Number)


@pytest.mark.parametrize(("func", "name"), OPENQASM_BUILTIN_FUNCTIONS)
def test_math_function_openqasm_emission(
    func: MathFunction,
    name: str,
    theta: FreeParameter,
) -> None:
    circuit = Circuit().rx(0, func(theta)).measure(0)
    qasm = circuit.to_ir(ir_type=IRType.OPENQASM).source
    assert f"rx({name}(theta)) q[0];" in qasm


@pytest.mark.parametrize(("sympy_fn", "openqasm_name"), SYMPY_TO_OPENQASM_PRINTER_CASES)
def test_sympy_function_openqasm_printer(
    sympy_fn: Any,
    openqasm_name: str,
    alpha: FreeParameter,
) -> None:
    if sympy_fn is sympy.Mod:
        expr = FreeParameterExpression(sympy_fn(alpha.expression, 2))
        expected = f"{openqasm_name}(alpha, 2)"
    else:
        expr = FreeParameterExpression(sympy_fn(alpha.expression))
        expected = f"{openqasm_name}(alpha)"
    assert str(expr) == expected
    assert repr(expr) == expected


@pytest.mark.parametrize(("expr_str", "expected_str"), STRING_FUNCTION_EXPRESSIONS)
def test_string_constructor_function_calls(expr_str: str, expected_str: str) -> None:
    expr = FreeParameterExpression(expr_str)
    assert str(expr) == expected_str
    assert repr(expr) == expected_str


@pytest.mark.parametrize(
    ("expr_str", "build_helper"),
    [
        ("sin(alpha)", lambda: sin(FreeParameter("alpha"))),
        ("arcsin(alpha)", lambda: arcsin(FreeParameter("alpha"))),
        ("mod(x, 2)", lambda: mod(FreeParameter("x"), 2)),
        (
            "sin(theta/2)**2 + cos(theta/2)**2",
            lambda: sin(FreeParameter("theta") / 2) ** 2 + cos(FreeParameter("theta") / 2) ** 2,
        ),
    ],
)
def test_string_constructor_round_trip(
    expr_str: str,
    build_helper: Callable[[], FreeParameterExpression],
) -> None:
    """String-parsed expressions should match helper-built equivalents."""
    from_string = FreeParameterExpression(expr_str)
    from_helper = build_helper()
    assert from_string == from_helper
    assert str(from_string) == str(from_helper)


def test_string_constructor_unknown_function() -> None:
    with pytest.raises(ValueError, match="Unknown function 'asin'"):
        FreeParameterExpression("asin(alpha)")


def test_math_helper_accepts_numeric_input() -> None:
    expr = sin(0.5)
    assert isinstance(expr, FreeParameterExpression)
    assert float(expr.expression) == pytest.approx(math.sin(0.5))


def test_mod_helper(theta: FreeParameter) -> None:
    expr_mod = mod(theta, 2)
    assert isinstance(expr_mod, FreeParameterExpression)
    assert str(expr_mod) == "mod(theta, 2)"
    assert expr_mod.subs({"theta": 5}) == 1.0


def test_openqasm_emission_arcsin(alpha: FreeParameter) -> None:
    circuit = Circuit().rx(0, arcsin(alpha)).measure(0)
    qasm = circuit.to_ir(ir_type=IRType.OPENQASM).source
    assert "rx(arcsin(alpha)) q[0];" in qasm


def test_local_simulator_arcsin(alpha: FreeParameter) -> None:
    """Issue reproducer: sympy.asin must serialize and execute as arcsin."""
    expr = FreeParameterExpression(sympy.asin(alpha.expression))
    circuit = Circuit().rx(0, expr).measure(0)
    result = LocalSimulator().run(circuit, inputs={"alpha": 0.5}, shots=10).result()
    assert len(result.measurements) == 10


def test_local_simulator_sin_cos_identity(alpha: FreeParameter) -> None:
    circuit = Circuit().rx(0, sin(alpha / 2) ** 2 + cos(alpha / 2) ** 2).measure(0)
    qasm = circuit.to_ir(ir_type=IRType.OPENQASM).source
    assert "sin(alpha/2)" in qasm
    assert "cos(alpha/2)" in qasm
    result = LocalSimulator().run(circuit, inputs={"alpha": math.pi}, shots=10).result()
    assert len(result.measurements) == 10


def test_unsupported_sympy_function() -> None:
    expr = FreeParameterExpression(sympy.Abs(FreeParameter("theta").expression))
    with pytest.raises(ValueError, match="No OpenQASM 3 equivalent for Abs"):
        str(expr)
