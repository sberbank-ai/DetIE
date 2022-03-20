# coding:utf-8
import stanza
from stanza import Document, Pipeline
from transformers import BertTokenizerFast

from .tags import (INV_UD_DEPREL_IDS, INV_UPOS_TAGS_IDS, UD_DEPREL_IDS,
                   UPOS_TAGS_IDS)


def parse_span_stanza(misc: str):
    start, end = misc.split("|")[:2]
    return int(start.split("=")[-1]), int(end.split("=")[-1])


def syntax_based_features_for_bpe(paragraph: str, syntax_model: Pipeline, tokenizer: BertTokenizerFast):
    # building a dependency tree
    parsed_paragraph: Document = syntax_model(paragraph)

    x = tokenizer([paragraph], return_tensors="pt", return_offsets_mapping=True)
    offsets_mapping = [tuple(row) for row in x["offset_mapping"].squeeze().detach().numpy()]

    parsed_sentence_heads, st_id2head, st_id2pos, st_id2deprel = [], {}, {}, {}
    mapped_bpe2st_id, sx2bpe_position = [], {0: None, len(offsets_mapping): None}

    for one_of_the_sentences in parsed_paragraph.sentences:

        last_token_offset = len(parsed_sentence_heads)
        sentence_text = one_of_the_sentences.text

        # yes, inefficient, but e.g. could be two spaces between sents
        char_sentence_offset = paragraph.find(sentence_text)
        last_char_in_sentence_offset = char_sentence_offset + len(sentence_text)

        if char_sentence_offset < 0:
            raise Exception("Could not find the sentence extracted by Stanza in text")

        # memorizing spans and mapping id2hea
        for t in one_of_the_sentences.tokens:
            head_orig_id = t.words[0].head
            head_id = None if head_orig_id == 0 else t.words[0].head + last_token_offset
            parsed_sentence_heads.append(parse_span_stanza(t.misc) + (t.id[0] + last_token_offset, head_id))
            st_id2head[t.id[0] + last_token_offset] = head_id
            st_id2pos[t.id[0] + last_token_offset] = t.words[0].upos
            st_id2deprel[t.id[0] + last_token_offset] = t.words[0].deprel

        current_syntax_span_id = last_token_offset
        assigned_bpe_ids = len(mapped_bpe2st_id)

        for bpe_span_id, (bpe_left, bpe_right) in enumerate(offsets_mapping[1:-1], start=1):

            if bpe_left >= last_char_in_sentence_offset:
                break

            if bpe_span_id < assigned_bpe_ids + 1:
                continue

            syntax_left, syntax_right, syntax_id, syntax_head_id = parsed_sentence_heads[current_syntax_span_id]

            # наша задача НАКРЫТЬ "синтаксическим" спаном БПЕ-шный
            # если левее, сдвигаем вправо, пока не наткнёмся на подходящий спан
            while syntax_left < bpe_left and current_syntax_span_id < len(parsed_sentence_heads) - 1:
                current_syntax_span_id += 1
                syntax_left, syntax_right, syntax_id, syntax_head_id = parsed_sentence_heads[current_syntax_span_id]

            # если правее, сдвигаем влево
            while syntax_left > bpe_left and current_syntax_span_id > last_token_offset:
                current_syntax_span_id = max(0, current_syntax_span_id - 1)
                syntax_left, syntax_right, syntax_id, syntax_head_id = parsed_sentence_heads[current_syntax_span_id]

            mapped_bpe2st_id.append(syntax_id)

            # saving the position of first occurrence of the syntax-based token_id in the BPE tokens list
            if not syntax_id in sx2bpe_position:
                sx2bpe_position[syntax_id] = bpe_span_id

    # for CLS
    heads, pos_tags, deprel_tags = [None], [None], [None]

    for syntax_id in mapped_bpe2st_id:
        try:
            syntax_head = st_id2head[syntax_id]
            if syntax_head in sx2bpe_position:
                heads.append(sx2bpe_position[syntax_head])
            else:
                # mapping to self as head because of abbreviations
                heads.append(None)
        except Exception as e:
            print(paragraph)
            raise e

        pos_tags.append(st_id2pos[syntax_id])
        deprel_tags.append(st_id2deprel[syntax_id])

    # for SEP
    heads.append(None)
    pos_tags.append(None)
    deprel_tags.append(None)

    assert len(mapped_bpe2st_id) + 2 == len(heads)
    assert len(deprel_tags) == len(heads) and len(pos_tags) == len(heads)

    return heads, [UPOS_TAGS_IDS[t] for t in pos_tags], [UD_DEPREL_IDS[t] for t in deprel_tags]


if __name__ == "__main__":
    TEST_TEXT = (
        "Tragedy We sing in the garden. Martie sings in the shower.  "
        "Aslan Maskhadov, "
        "the separatist  president of Chechnya (ChRI),  opposed the invasion of Dagestan, "
        "and offered a crackdown on the renegade warlords."
    )

    # stanza.download("en")
    pipeline = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    bpe_tokens = tokenizer(TEST_TEXT).tokens()

    for s in pipeline(TEST_TEXT).sentences:
        tokens = s.tokens
        print([t.text for t in tokens])
        tok2head = {t.id[0]: t.words[0].head for t in tokens}
        for t in tokens:
            head_id = tok2head[t.id[0]]
            print("%10s <- (%2d) %10s" % (t.text, tok2head[t.id[0]], tokens[head_id - 1].text if head_id > 0 else None))
        print("---------")

    print()
    heads, pos_tags, deprel_tags = syntax_based_features_for_bpe(TEST_TEXT, pipeline, tokenizer)
    print(heads, pos_tags, deprel_tags, sep="\n")

    for t, (h, (p_t, d_t)) in zip(bpe_tokens, zip(heads, zip(pos_tags, deprel_tags))):
        print(
            "%10s [%5s][%6s] <- %10s"
            % (t, INV_UPOS_TAGS_IDS[p_t], INV_UD_DEPREL_IDS[d_t], bpe_tokens[h] if h is not None else "None")
        )
