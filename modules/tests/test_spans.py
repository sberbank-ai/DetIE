# -*- coding: utf-8 -*-

import pytest

from modules.data.spans import find_spans


@pytest.mark.parametrize(
    "text,patterns,out",
    [
        [
            "As the regiment is a proper noun , Toronto Maple Leafs plural is Maple Leafs ( not Maple Leaves ) .",
            ["Toronto Maple Leafs plural", "is", "Maple Leafs As the regiment is a proper noun"],
            [[[35, 61]], [[16, 18]], [[0, 16], [18, 32], [65, 77], [62, 64]]],
        ],
        [
            # "In 1996 , Stavro took on Larry Tanenbaum , cofounder of Toronto 's new National Basketball Association ( NBA ) team , the Toronto Raptors , as a partner .",
            # ["Toronto 's new National Basketball Association NBA team", 'is', 'the Toronto Raptors'],
            # []
            "Within a year of Anthony Trollope marriage , Anthony Trollope finished that work .",
            ["Anthony Trollope", "finished", "that work Within a year of Anthony Trollope marriage"],
            [[[17, 33]], [[62, 70]], [[44, 62], [0, 16], [71, 81], [34, 42]]],
        ],
    ],
)
def test_find_spans(text, patterns, out):

    print(find_spans(text, patterns))

    for spans in find_spans(text, patterns):
        print(*[text[span[0] : span[1]] for span in spans], sep="\n")

    print()
    assert find_spans(text, patterns) == out
