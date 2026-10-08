"""Remove declared thought text only; never repair/select code by test success."""
import ast
import re

PROTOCOL = 'code-final-answer-after-thought-v4'
FENCE_RE = re.compile(r'```[^\n]*\n.*?```', re.DOTALL)


def final_code_text(completion):
    # A valid Python literal containing </think> is code, not a reasoning tag.
    try:
        ast.parse(completion.strip())
        return completion.strip()
    except (SyntaxError, ValueError):
        pass
    blocks = [(m.start(), m.end()) for m in FENCE_RE.finditer(completion)]
    closers = [m for m in re.finditer(r'</think>', completion)
               if not any(a <= m.start() < b for a,b in blocks)]
    if closers:
        return completion[closers[-1].end():].strip()
    # Legacy body-only/plain-code outputs are left unchanged. An unfinished
    # thought is not mined for a plausible function or a passing intermediate.
    return completion.strip()


def self_test():
    good = 'def f(x):\n    return x + 1'
    cases = [
        (good, good),
        ('    return x + 1', 'return x + 1'),
        ('Earlier prose\n</think>\n'+good, good),
        ('<think>draft```python\ndef f(x): return x - 1\n```\n</think>\n'+good, good),
        ('<think>draft\n</think>\n```python\n'+good+'\n```', '```python\n'+good+'\n```'),
        ('def f():\n    return "</think>"', 'def f():\n    return "</think>"'),
        ('```python\ndef f():\n    return "</think>"\n```', '```python\ndef f():\n    return "</think>"\n```'),
        ('draft\n</think>\n```python\ndef f(): return "</think>"\n```', '```python\ndef f(): return "</think>"\n```'),
        ('<think>unfinished draft with def f(x): return x', '<think>unfinished draft with def f(x): return x'),
        ('</think>\ndef f(x):\n    return', 'def f(x):\n    return'),
        ('</think>\n', ''),
    ]
    for raw, expected in cases:
        assert final_code_text(raw) == expected, (raw, final_code_text(raw), expected)
    print('FINAL_CODE_TEXT_TEST_PASS', len(cases), flush=True)


if __name__ == '__main__':
    self_test()
