# coding: utf-8
import torch
from transformers import BatchEncoding, GPT2LMHeadModel, GPT2TokenizerFast


class PretrainedLMScorer(object):
    def __init__(self, huggingface_model_id: str = "distilgpt2", stride: int = 4):

        self.model_id = huggingface_model_id

        if "gpt2" in huggingface_model_id:
            self.model = GPT2LMHeadModel.from_pretrained(self.model_id)  # .to(device)
            self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_id)
        else:
            raise Exception("Model [%s] is not supported" % huggingface_model_id)

        self.max_length = self.model.config.n_positions
        self.stride = stride

    def eval(self, text: str):

        self.encodings = self.tokenizer(text, return_tensors="pt")  # type: BatchEncoding
        lls = []
        end_loc = 0

        for i in range(0, self.encodings.input_ids.size(1), self.stride):

            # context start and context end
            begin_loc = max(i + self.stride - self.max_length, 0)
            end_loc = min(i + self.stride, self.encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # context tokens columns -- in a batch
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc]  # .to(device)

            # tokens we compute cond. probability for
            target_ids = input_ids.clone()

            # excluding all the summands but the last ones from the future likelihood sum
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len

            lls.append(log_likelihood)

        return torch.exp(torch.stack(lls).sum() / end_loc).item()


if __name__ == "__main__":

    import stanza

    # stanza.download('en')
    nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos")
    doc = nlp("is performance of")
    print(
        *[
            f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}'
            for sent in doc.sentences
            for word in sent.words
        ],
        sep="\n",
    )
    print()
