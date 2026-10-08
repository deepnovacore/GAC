"""Versioned, CPU-only SciBench numeric scoring; never eval/sympify model text.

v2: balanced final boxes, bounded arithmetic, explicit units and scale factors.
Bare numbers are in the requested units. Explicit units must be compatible.
Unsupported/ambiguous answers fail closed with a reason, not a guessed number.
Tolerance is 1% relative (absolute 1e-12 ONLY for a zero reference).
This is a protocol revision, not a silent replacement of historical scores.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import asdict, dataclass

PROTOCOL = "scibench-numeric-units-v2"


def final_box(text: str) -> tuple[str, int, int] | None:
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    m = matches[-1]
    start, pos, depth = m.end(), m.end(), 1
    while pos < len(text) and depth:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return (text[start:pos - 1].strip(), m.start(), pos) if not depth else None


def _group(s: str, pos: int) -> tuple[str, int]:
    while pos < len(s) and s[pos].isspace():
        pos += 1
    if pos >= len(s) or s[pos] != "{":
        raise ValueError("expected_braced_argument")
    start, depth = pos + 1, 1
    pos += 1
    while pos < len(s) and depth:
        depth += (s[pos] == "{") - (s[pos] == "}")
        pos += 1
    if depth:
        raise ValueError("unbalanced_argument")
    return s[start:pos - 1], pos


def latex_plain(s: str, depth: int = 0) -> str:
    if depth > 12 or len(s) > 2000:
        raise ValueError("expression_limit")
    out, pos = [], 0
    while pos < len(s):
        m = re.match(r"\\(dfrac|tfrac|frac|sqrt|mathrm|text|textrm|textbf|mathbf)\b", s[pos:])
        if not m:
            out.append(s[pos]); pos += 1; continue
        cmd = m.group(1)
        arg, end = _group(s, pos + m.end())
        arg = latex_plain(arg, depth + 1)
        if cmd in {"frac", "dfrac", "tfrac"}:
            den, end = _group(s, end)
            out.append("((" + arg + ")/(" + latex_plain(den, depth + 1) + "))")
        elif cmd == "sqrt":
            out.append("((" + arg + ")**0.5)")
        else:
            out.append(" " + arg + " ")
        pos = end
    value = "".join(out).replace("−", "-").replace("×", "*")
    value = value.replace(r"\$", " USD ").replace("$", "")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = re.sub(r"\\(?:[,;!: ]|quad\b|qquad\b)", " ", value).replace("~", " ")
    for a, b in [(r"\times", "*"), (r"\cdot", "*"), ("·", "*"),
                 (r"\pi", "pi"), ("π", "pi"), (r"\mu", "u"), ("µ", "u"),
                 (r"\Omega", "Ohm"), (r"\%", "%"), (r"\circ", "deg"), ("°", "deg")]:
        value = value.replace(a, b)
    return value.strip()


def numeric(expression: str) -> float:
    s = latex_plain(str(expression)).strip().rstrip(".")
    s = re.sub(r"(?<![\d,])\d{1,3}(?:,\d{3})+(?![\d,])", lambda m: m[0].replace(",", ""), s)
    s = s.replace("{", "(").replace("}", ")").replace("^", "**")
    s = re.sub(r"(?<=\d)(?=pi\b)", "*", s)
    if not s or len(s) > 256:
        raise ValueError("numeric_expression_limit")
    tree = ast.parse(s, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("numeric_expression_limit")

    def calc(n: ast.AST, depth: int = 0) -> float:
        if depth > 20:
            raise ValueError("numeric_depth_limit")
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            v = float(n.value)
        elif isinstance(n, ast.Name) and n.id in {"pi", "e"}:
            v = math.pi if n.id == "pi" else math.e
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = calc(n.operand, depth + 1) * (-1 if isinstance(n.op, ast.USub) else 1)
        elif isinstance(n, ast.BinOp):
            a, b = calc(n.left, depth + 1), calc(n.right, depth + 1)
            if isinstance(n.op, ast.Add): v = a + b
            elif isinstance(n.op, ast.Sub): v = a - b
            elif isinstance(n.op, ast.Mult): v = a * b
            elif isinstance(n.op, ast.Div): v = a / b
            elif isinstance(n.op, ast.Pow) and abs(b) <= 100 and abs(a) <= 1e100: v = a ** b
            else: raise ValueError("unsupported_arithmetic")
        else:
            raise ValueError("unsupported_arithmetic")
        if not isinstance(v, (float, int)) or not math.isfinite(v) or abs(v) > 1e200:
            raise ValueError("non_finite_arithmetic")
        return v
    return calc(tree.body)


# Same-named dimensions plus SI prefixes. No guessed N*m -> J, Celsius offsets,
# or prose/vector conversion. Unsupported representations stay reviewable.
_BASE = set("m s g mol K J cal Pa N C V A W Hz T F H Ohm eV atm bar L rad deg degC degN USD percent ft lb in mi h min day yr month slug AU Torr MeV D Angstrom photon electron e dB".split())
_ALIASES = {"seconds": "s", "second": "s", "sec": "s", "hours": "h", "hour": "h",
            "days": "day", "years": "yr", "year": "yr", "Year": "yr", "months": "month",
            "slugs": "slug", "photons": "photon", "electrons": "electron", "%": "percent", "Å": "Angstrom"}
_PREFIXES = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
             "c": 1e-2, "d": 1e-1, "k": 1e3, "M": 1e6, "G": 1e9}
_PREFIXABLE = set("m s g J cal Pa N C V A W Hz T F H Ohm eV L".split())


def units(expression: str) -> tuple[float, tuple[tuple[str, int], ...]]:
    s = latex_plain(expression).strip().rstrip(".")
    s = re.sub(r"\bu\s+(?=[A-Za-z])", "u", s)
    s = re.sub(r"\^\s*\{?deg\}?", "deg", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\bdeg\s+([CN])\b", r"deg\1", s)
    s = re.sub(r"ft\s*-\s*lb", "ft*lb", s)
    # Empty LaTeX groups in a few degree labels carry no dimension.
    s = re.sub(r"\(\s*\)", "", s)
    tokens = re.findall(r"[A-Za-zÅ%]+|[+-]?\d+|[*/^()]", s)
    if len(tokens) > 256:
        raise ValueError("unit_expression_limit")
    if re.sub(r"\s+", "", s) != "".join(tokens):
        raise ValueError("unsupported_unit_syntax")
    if not tokens:
        return 1., ()
    pos = 0

    def product(stop: bool = False, depth: int = 0):
        nonlocal pos
        if depth > 20:
            raise ValueError("unit_depth_limit")
        scale, dims, divide, count = 1., {}, False, 0
        while pos < len(tokens) and tokens[pos] != ")":
            t = tokens[pos]
            if t in {"*", "/"}:
                if not count: raise ValueError("unit_operator")
                divide = t == "/"; pos += 1
                if pos == len(tokens) or tokens[pos] in {"*", "/", ")"}: raise ValueError("unit_operator")
                continue
            pos += 1
            if t == "(":
                factor, term = product(True, depth + 1)
            elif t == "10":
                factor, term = 10., {}
            elif t == "1":
                factor, term = 1., {}
            else:
                atom = _ALIASES.get(t, t)
                factor = 1.
                if atom not in _BASE:
                    if len(atom) > 1 and atom[0] in _PREFIXES and atom[1:] in _PREFIXABLE:
                        factor, atom = _PREFIXES[atom[0]], atom[1:]
                    else: raise ValueError("unknown_unit:" + atom)
                term = {atom: 1}
            power = 1
            if pos < len(tokens) and tokens[pos] == "^":
                pos += 1
                grouped = pos < len(tokens) and tokens[pos] == "("
                if grouped: pos += 1
                if pos >= len(tokens) or not re.fullmatch(r"[+-]?\d+", tokens[pos]): raise ValueError("unit_exponent")
                power = int(tokens[pos]); pos += 1
                if abs(power) > 100: raise ValueError("unit_exponent")
                if grouped:
                    if pos >= len(tokens) or tokens[pos] != ")": raise ValueError("unit_exponent")
                    pos += 1
            if divide: power = -power
            scale *= factor ** power
            for k, v in term.items(): dims[k] = dims.get(k, 0) + v * power
            divide = False; count += 1
        if stop:
            if pos >= len(tokens) or tokens[pos] != ")": raise ValueError("unit_parentheses")
            pos += 1
        return scale, dims

    scale, dims = product()
    if pos != len(tokens) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid_unit")
    return scale, tuple(sorted((k, v) for k, v in dims.items() if v))


def quantity(answer: str, expected_unit: str | None) -> tuple[float, str]:
    if len(answer) > 2000:
        raise ValueError("answer_too_long")
    # An optional variable label, not an arbitrary sequence of asserted answers.
    parts = re.split(r"\\approx|≈|(?<![!<>])=", answer)
    if len(parts) > 4:
        raise ValueError("ambiguous_answer_chain")
    if len(parts) >= 2:
        terminal, status = quantity(parts[-1].strip(), expected_unit)
        for i, part in enumerate(parts[:-1]):
            label = part.strip().replace("$", "")
            is_label = len(label) < 120 and re.fullmatch(r"(?:\\[A-Za-z]+|[A-Za-z])[A-Za-z0-9_{}\\\s^,().-]*", label)
            if i == 0 and is_label:
                continue
            # Verify each numerical equality/approximation, not just the last
            # number. Unsupported symbolic intermediate expressions fail closed.
            value, _ = quantity(part.strip(), expected_unit)
            if not math.isclose(value, terminal, rel_tol=.01, abs_tol=1e-12 if terminal == 0 else 0.):
                raise ValueError("inconsistent_answer_chain")
        return terminal, status
    if answer.lstrip().startswith(r"\$"):
        answer = answer.lstrip()[2:] + " USD"
    # Unit metadata need not parse for an unannotated scalar: it is already
    # expressed in the question's requested scale, not an inferred SI value.
    try:
        return numeric(answer), "implicit_requested_unit"
    except (ValueError, SyntaxError, ArithmeticError, TypeError):
        pass
    expected_scale, expected_dims = units(expected_unit or "")
    plain = latex_plain(answer).strip().rstrip(".")
    candidates = []
    for i in range(1, len(plain)):
        # Never split a scalar's digits: 131 K is not 13 * (1 K).
        if plain[i - 1] in ".0123456789" and plain[i] in ".0123456789":
            continue
        left, right = plain[:i].strip(), plain[i:].strip()
        if not left or not right: continue
        try:
            number = numeric(left)
            scale, dims = units(right)
        except (ValueError, SyntaxError, ArithmeticError, TypeError):
            continue
        if dims == expected_dims:
            candidates.append(number * scale / expected_scale)
    if not candidates:
        raise ValueError("unparsed_or_incompatible_unit")
    if any(not math.isfinite(n) for n in candidates) or any(not math.isclose(n, candidates[0], rel_tol=1e-12, abs_tol=0.) for n in candidates):
        raise ValueError("ambiguous_quantity")
    return candidates[0], "explicit_compatible_unit"


@dataclass
class Score:
    correct: bool = False
    reason: str = "unparsed"
    prediction: float | None = None
    reference: float | None = None
    boxed: str | None = None
    unit_status: str | None = None
    protocol: str = PROTOCOL

    def to_dict(self):
        return asdict(self)


def score_science(completion: str, gold: str, unit: str | None = None) -> Score:
    result = Score()
    box = final_box(completion)
    if box is None:
        result.reason = "missing_or_unbalanced_final_box"; return result
    result.boxed = box[0]
    try:
        result.reference = numeric(str(gold))
        answer = box[0]
        suffix = completion[box[2]:].strip().strip("$").strip().rstrip(".")
        # A unit written immediately outside the final box is still a unit.
        # Do not turn `boxed{50.7} Pa` into a unit-free answer in atm.
        if suffix and not re.search(r"[\n.!?]", suffix):
            try:
                units(suffix)
                numeric(answer)
            except (ValueError, SyntaxError, ArithmeticError, TypeError):
                pass
            else:
                answer += " " + suffix
        result.prediction, result.unit_status = quantity(answer, unit)
        tolerance = abs(result.reference) * .01 if result.reference != 0 else 1e-12
        result.correct = abs(result.prediction - result.reference) <= tolerance
        result.reason = "numeric_match" if result.correct else "numeric_mismatch"
    except (ValueError, SyntaxError, ArithmeticError, TypeError) as exc:
        result.reason = "parse_rejected:" + str(exc)[:100]
    return result
