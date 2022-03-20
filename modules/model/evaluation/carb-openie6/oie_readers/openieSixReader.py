# coding: utf-8

from oie_readers.extraction import Extraction
from oie_readers.oieReader import OieReader


class OpenieSixReader(OieReader):
    def __init__(self):
        self.name = "OpenIE-6"

    def read(self, fn: str, includeNominal: bool = False):

        d = {}

        with open(fn, "r+", encoding="utf-8") as rf:
            inside = False
            current_data = {}

            for line in rf:
                if line.strip():
                    if not inside:
                        sentence = line.strip()
                        current_data["sentence"] = sentence
                        current_data["triples"] = []
                    else:
                        splitted = line.split(": ")
                        confidence = float(splitted[0])
                        triple = (": ".join(splitted[1:])).strip().strip(")").strip("(").split("; ")
                        third = " ".join(triple[2:])

                        current_data["triples"].append((triple[0], triple[1], third, confidence))

                    inside = True
                else:
                    inside = False

                    for arg1, rel, arg2, conf in current_data["triples"]:
                        curExtraction = Extraction(
                            pred=rel, head_pred_index=-1, sent=current_data["sentence"], confidence=float(conf)
                        )
                        curExtraction.addArg(arg1)
                        curExtraction.addArg(arg2)
                        d[current_data["sentence"]] = d.get(current_data["sentence"], []) + [curExtraction]

        self.oie = d


if __name__ == "__main__":
    reader = OpenieSixReader()
    reader.read("./system_outputs/test/openie6_output.txt")

    for k, v in reader.oie.items():
        print(k)
        for vv in v:
            print(vv.pred, ">", vv.args)
        print()
