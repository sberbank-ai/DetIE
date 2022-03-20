# -*- coding: utf-8 -*-

import re
from typing import *

import pytest

from modules.data.spans import STOP_TOKENS, find_spans


@pytest.fixture
def texts_and_patterns() -> Tuple[List[str], List[str]]:
    return [
        "Justice Hugo Black stated : By this time , four states had a minimum voting age below 21 .",
        "The legislation embodying the recommendation was reportedly approved by the Massachusetts House of Representatives on a vote of 138 to 29 .",
        "The legislation embodying the recommendation was reportedly approved by the Massachusetts House of Representatives on a vote of 138 to 29 .",
    ], [
        " Justice a minimum voting age below 21 By this time ",
        "The legislation embodying the recommendation",
        "by the Massachusetts House of Representatives on a vote of 138 to 29",
    ]


def filter_tokens(string):
    return re.sub(f"[{STOP_TOKENS}]*", "", string)


def test_find_spans(texts_and_patterns: pytest.fixture):

    for i, (text, pattern) in enumerate(zip(*texts_and_patterns)):
        spans = find_spans(text, pattern)

        for span in spans:
            assert text[span[0] : span[1]] in pattern

        assert len(filter_tokens(pattern)) == len(filter_tokens("".join([text[span[0] : span[1]] for span in spans])))

        if 1 <= i <= 3:
            assert len(spans) == 1
            span = spans[0]
            assert filter_tokens(text[span[0] : span[1]]) == filter_tokens(pattern)
