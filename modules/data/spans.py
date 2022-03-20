# -*- coding: utf-8 -*-

from difflib import SequenceMatcher
from string import punctuation, whitespace

STOP_TOKENS = punctuation + whitespace


def find_spans(text, patterns, steps=5):

    matcher = SequenceMatcher(a=text)
    out, spans = [], [[0, len(text)]]

    for pattern in patterns:
        out.append([])
        for _ in range(steps):
            pattern = pattern.strip()
            if not pattern:
                break
            matcher.set_seq2(pattern)

            if not spans:
                break

            # For all spans find longest matches with the pattern
            matches = []
            for span in spans:
                matches.append(matcher.find_longest_match(span[0], span[1], 0, len(pattern)))

            # Find the longest match among all spans
            matched_span_idx = max(range(len(matches)), key=lambda i: matches[i].size)
            largest_match = matches[matched_span_idx]

            if not largest_match.size:
                break

            # Cut the match out of the span splitting it into two parts so as not to allow duplicate matches further
            matched_span = spans[matched_span_idx]
            span_replacement = []
            for left, right in (
                (matched_span[0], largest_match.a),
                (largest_match.a + largest_match.size, matched_span[1]),
            ):
                if text[left:right].strip(STOP_TOKENS):
                    span_replacement.append([left, right])
            spans[matched_span_idx : matched_span_idx + 1] = span_replacement

            pattern = pattern[: largest_match.b] + pattern[largest_match.b + largest_match.size :]
            out[-1].append([largest_match.a, largest_match.a + largest_match.size])

    # # We omit missing "is" in rare cases
    # for i, (item, pattern) in enumerate(zip(out, patterns)):
    #     if not item and pattern:
    #         print(f'Missing item {i}')
    #         print(text)
    #         print(pattern)

    return out
