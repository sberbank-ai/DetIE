#!/usr/bin/env bash
set -e

python3 -c 'import nltk; nltk.download("stopwords")'
python3 -m pip install docopt

for dataset in "test"; do

  echo $dataset
  gold_fp=./data/gold/$dataset.tsv

  for tuple in "detie243","ollie" "detie263","ollie" "detie243conj","ollie" "detie263conj","ollie"; do
    IFS=","
    set -- $tuple
    echo "Evaluation of $1 ($2 format) -------------------"
    echo "carb(s,s)"
    python3 carb.py --$2 systems_output/$1_output.txt --gold $gold_fp --out /dev/null --single_match
    echo "carb(s,m)"
    python3 carb.py --$2 systems_output/$1_output.txt --gold $gold_fp --out /dev/null
    echo "oie16"
    python3  -W ignore oie16.py --$2 systems_output/$1_output.txt  --gold $gold_fp --out /dev/null
    echo "wire57"
    if [[ "$2" = "allennlp" ]]
    then
      python3 wire57_evaluation.py --system systems_output/$1_output.txt --gold data/${dataset}_gold_allennlp_format.txt
    elif [[ "$2" = "ollie" ]]
    then
      python3 oie_tabbed_to_allennlp.py --inp systems_output/$1_output.txt --out systems_output/$1_output.allennlp
      python3 wire57_evaluation.py --system systems_output/$1_output.allennlp --gold data/${dataset}_gold_allennlp_format.txt
    else
      echo "Cannot process the file of format not equal to 'allennlp' or 'ollie': $2"
    fi
    echo ""
  done
done