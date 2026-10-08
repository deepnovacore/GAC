"""Gold-independent BBH choice-label extraction, protocol v3.

No semantic judge, searching for the gold label, or reasoning-based rescue.
An explicit final choice may be bare, Markdown/LaTeX wrapped, or accompanied
by its exact option text. Ambiguity and conflicting letter/text fail closed.
Numeric object-counting and every other task's scorer remain unchanged.
"""
import re

PROTOCOL = 'bbh-explicit-choice-label-v3'
OPTION_RE = re.compile(r'^\(([A-Z])\)\s+(.+?)\s*$', re.MULTILINE)
MARKER_RE = re.compile(
    r'\b(?:the\s+(?:(?:correct|final)\s+)?answer\s+is|'
    r'(?:final\s+)?answer\s*:)\s*[:\-]?\s*([^\n]+)', re.IGNORECASE)


def plain(text):
    text = re.sub(r'\\(?:text|mathrm|mathbf|mathit)\s*\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\boxed\s*\{([^{}]*)\}', r'\1', text)
    return re.sub(r'\s+', ' ', text.replace('**', '').replace('`', '').strip()).strip('$ .:')


def norm(text):
    return plain(text).casefold()


def options_from_prompt(prompt):
    pairs = OPTION_RE.findall(prompt)
    result = dict(pairs)
    if len(pairs) != len(result):
        raise ValueError('Duplicate BBH option letters')
    if len(result) not in (3, 5, 7) or set(result) != set('ABCDEFG'[:len(result)]):
        raise ValueError('Missing, duplicate, or non-contiguous BBH options')
    if len({norm(v) for v in result.values()}) != len(result):
        raise ValueError('Non-unique BBH option texts')
    return result


def boxed_candidates(text):
    for match in re.finditer(r'\\boxed\s*\{', text):
        start = match.end()
        depth = 1
        for end in range(start, len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                yield match.start(), text[start:end]
                break


def candidate_label(candidate, options):
    value = plain(candidate)
    # A standalone option text must exactly match one of the presented options.
    exact = [k for k, v in options.items() if norm(value) == norm(v)]
    if len(exact) == 1:
        return exact[0]
    match = re.fullmatch(r'(?:(?:option|choice)\s+)?\(?([A-Z])\)?(.*)', value,
                         flags=re.IGNORECASE)
    if match is None or match[1].upper() not in options:
        return None
    label, suffix = match[1].upper(), match[2].strip(' .:-)')
    if not suffix or norm(suffix) == norm(options[label]):
        return label
    # Do not accept "A or B", guessed initials, or "(D) Sam" for option D=Ophelia.
    return None


def extract_choice(completion, options):
    # The generation prompt already contains <think>; completions may have only
    # the closing tag. Never let an earlier reasoning answer override a final one.
    final = completion.rsplit('</think>', 1)[-1]
    if '<think>' in final:
        return None
    markers = list(MARKER_RE.finditer(final))
    # A boxed letter embedded in "The answer is \\boxed{A} or B" must not
    # supersede the ambiguous containing marker merely because it starts later.
    candidates = [(start, value) for start, value in boxed_candidates(final)
                  if not any(m.start() <= start < m.end() for m in markers)]
    candidates += [(m.start(), m[1]) for m in markers]
    if candidates:
        _, answer = max(candidates, key=lambda x: x[0])
        return candidate_label(answer, options)
    lines = [line.strip() for line in final.splitlines() if line.strip()]
    return candidate_label(lines[-1], options) if lines else None


def self_test():
    options = {'A': 'Helga', 'B': 'Karl', 'C': 'Melissa', 'D': 'Ophelia', 'E': 'Sam'}
    cases = [
        ('The answer is (D) Ophelia.', 'D'),
        ('**The answer is (D)**.', 'D'),
        ('The answer is **D**.', 'D'),
        ('The answer is D) Ophelia.', 'D'),
        ('(D) Ophelia', 'D'),
        ('Answer: D', 'D'),
        ('Final answer: (D).', 'D'),
        ('\\boxed{D}', 'D'),
        ('\\boxed{\\text{D}}', 'D'),
        ('The answer is Ophelia.', 'D'),
        ('\\boxed{D}\nThe answer is (A) Helga.', 'A'),
        ('The answer is A\n</think>\nThe answer is D.', 'D'),
        ('The answer is A\n</think>\nI do not know.', None),
        ('<think>The answer is D', None),
        ('The answer is D or A.', None),
        ('The answer is \\boxed{D} or A.', None),
        ('The answer is \\boxed{\\text{D}}.', 'D'),
        ('The answer is (D) Sam.', None),
        ('\\boxed{O}', None),
        ('Ophelia (O).', None),
        ('The answer is D, maybe A.', None),
        ('All options are A B C D E.', None),
        ('', None),
        ('\\boxed{\\text{D}', None),
    ]
    for text, expected in cases:
        assert extract_choice(text, options) == expected, (text, expected, extract_choice(text, options))
    prompt = '\n'.join(f'({k}) {v}' for k, v in options.items())
    assert options_from_prompt(prompt) == options
    print('BBH_CHOICE_PROTOCOL_TEST_PASS', len(cases) + 1)


if __name__ == '__main__':
    self_test()
