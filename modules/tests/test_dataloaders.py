# -*- coding: utf-8 -*-

from collections import defaultdict

import pytest
from transformers import AutoTokenizer

from modules.data.spans import find_spans
from modules.model.dataloaders import (CollateFn, find_max_lens,
                                       pad_nested_lists)


@pytest.mark.parametrize(
    "nested_lists, gt_lens",
    [
        [[[1], [1, 2]], [2]],
        [
            [
                [
                    [[1, 2, 3, 4], [1], [1, 2, 3, 4, 5]],
                    [[1, 2, 3, 4], [1], [1, 2, 3, 4, 5]],
                    [[1, 2, 3, 4], [1], [1, 2, 3, 4, 5]],
                ],
                [
                    [[1, 2, 3, 4], [1], [1, 2, 3]],
                    [[1, 2, 3, 4], [1], [1, 2, 3]],
                    [[1, 2, 3, 4], [1], [1, 2, 3], [5]],
                    [[1, 2, 3, 4], [1], [1, 2, 3], [5]],
                    [[1, 2, 3, 4], [1], [1, 2, 3], [5]],
                    [[1, 2, 3, 4], [1], [1, 2, 3], [5]],
                ],
            ],
            [6, 4, 5],
        ],
        [
            [
                [
                    [1, 2, 3, 4],
                    [1, 2, 3, 4, 5, 6],
                ],
                [
                    [1, 2, 3, 4],
                    [1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4, 5, 6, 7],
                ],
            ],
            [4, 7],
        ],
    ],
)
def test_pad_nested_lists(nested_lists, gt_lens):

    lens = defaultdict(int)
    find_max_lens(nested_lists, lens)

    for key, value in lens.items():
        assert value == gt_lens[key]

    tensor = pad_nested_lists(nested_lists)
    assert list(tensor.shape[1:]) == gt_lens


def form_batch(texts, extractions):

    indices = []

    for text, text_extractions in zip(texts, extractions):
        indices.append([[find_spans(text, item) for item in extraction] for extraction in text_extractions])

    return indices


def test_collate_fn_multiple_spans():

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    collate_fn = CollateFn(tokenizer, use_syntax_features=False, word_dropout=0.0, multiple_spans=True)

    texts = [
        "He was really hard-working nice person.",
        "I have to be in some other place",
        "Justice Hugo Black stated : By this time , four states had a minimum voting age below 21 .",
        "On March 10 , 1971 , the Senate voted 94 -- 0 in favor of proposing a Constitutional amendment to guarantee that the voting age could not be higher than 18 .	",
    ]

    extractions = [
        [["He", "was", "hard-working person"], ["He", "was", "really nice"]],
        [["I", "have to be", "in some other place"]],
        [["four states ", " had  ", " a minimum voting age below 21 By this time"]],
        [["the Senate ", " voted ", "94 -- 0 in favor of proposing a Constitutional amendment On March 10 "]],
    ]
    gt = [
        [
            [0, 1, 2, 0, 3, 3, 3, 0, 0, 3, 0, 0],
            [0, 1, 2, 3, 0, 0, 0, 3, 3, 0, 0, 0],
        ],
        [
            [0, 1, 2, 2, 2, 3, 3, 3, 3, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 0, 0],
        ],
        [
            [
                0,
                3,
                3,
                3,
                0,
                0,
                0,
                1,
                1,
                2,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ],
    ]
    indices = form_batch(texts, extractions)
    batch = list(zip(texts, indices, *[[None] * len(texts)] * 3))
    assert len(batch) == len(texts) == len(extractions)
    tokenized, masks, _ = collate_fn(batch)

    gt[0][0] = gt[0][0] + [0] * (tokenized["input_ids"].shape[1] - len(gt[0][0]))
    gt = pad_nested_lists(gt)

    for tokens, seq_mask, seq_gt in zip(tokenized["input_ids"], masks, gt):

        for rel_mask, rel_gt in zip(seq_mask.transpose(0, 1), seq_gt):
            for token, token_mask in zip(tokens, rel_mask.to(int).argmax(-1)):
                print(token_mask.tolist(), tokenizer.decode(token.unsqueeze(0)))

            assert len(rel_gt) == len(rel_mask)
            assert rel_gt.tolist() == rel_mask.to(int).argmax(-1).tolist()
