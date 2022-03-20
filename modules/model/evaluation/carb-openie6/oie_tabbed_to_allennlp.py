# coding: utf-8

import argparse

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--conj", type=str)  # original\n(split sentences) file for mapping

    return parser


parser = parse_args()
args = parser.parse_args()

in_f = open(args.inp, "r")

# reading all lines, removing headers
examples = [l.strip() for l in in_f.readlines()][1:]

conj_mapping = dict()
conj_mapping_values = set()

if args.conj:
    content = open(args.conj).read()

    for example in content.split("\n\n"):
        for i, line in enumerate(example.strip("\n").split("\n")):
            if i == 0:
                orig_sentence = line
            else:
                conj_mapping[line] = orig_sentence

    conj_mapping_values = conj_mapping.values()

out_f = open(args.out, "w")

for example_id, example in tqdm(enumerate(examples)):
    if not example:
        continue

    columns = example.split("\t")
    sentence = columns[6]

    if sentence in conj_mapping_values:  # ignore extractions of original sentence
        continue

    if sentence in conj_mapping:  # replace split sentence with original sentence
        sentence = conj_mapping[sentence]

    confidence = columns[0]

    # for extraction in extractions:
    #     confidence = extraction.split(' ')[0].strip(':')
    #     extraction = ' '.join(extraction.split(' ')[1:])
    #     if 'Context' in extraction:
    #         extraction = ' '.join(extraction.split(':')[1:])
    #     fields = extraction.split(';')
    # try:
    #     match = re.search('(.*):\((.*); (.*); (.*)\)', extraction)
    #     # includes context
    #     confidence = match.group(1).strip()
    #     confidence = confidence.split()[0]
    # except:
    #     match = re.search('(\d.\d\d) \((.*); (.*); (.*)\)', extraction)
    #     confidence = match.group(1).strip()

    subject = columns[1].strip()  # remove opening bracket
    relation = columns[2].strip()
    object = columns[3].strip()
    out_f.write(f"{sentence}\t<arg1> {subject} </arg1> <rel> {relation} </rel> <arg2> {object} </arg2>\t{confidence}\n")

out_f.close()
