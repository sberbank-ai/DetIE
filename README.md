# DetIE: Multilingual Open Information Extraction Inspired by Object Detection

This repository contains the code for the paper 
[DetIE: Multilingual Open Information Extraction Inspired by Object Detection](https://www.aaai.org/AAAI22Papers/AAAI-8073.VasilkovskyM.pdf)
by Michael Vasilkovsky, Anton Alekseev, Valentin Malykh, Ilya Shenbin, Elena Tutubalina, 
Dmitriy Salikhov, Mikhail Stepnov, Andrei Chertok and Sergey Nikolenko.

## Disclaimers

All the results have been obtained using V100 GPU with CUDA 10.1. 

## Preparations

Download the files bundle from 
[here](https://drive.google.com/drive/folders/1SGeQWcFwmL4BaMbCTxVw5-oU69vPW_d-?usp=sharing). Each of them 
should be put into the corresponding directory:
1. folder `version_243` (DetIE_LSOIE) should be copied to: `results/logs/default/version_243`;
2. folder `version_263` (DetIE_IMoJIE) should be copied to: `results/logs/default/version_263`;
3. files `imojie_train_pattern.json`, `lsoie_test10.json` and `lsoie_train10.json` should be copied to `data/wikidata`.

We suggest that you use the provided [Dockerfile](/Dockerfile) to deal with all the dependencies of this project.

E. g. clone this repository, then
```bash
cd DetIE/
docker build -t detie .
nvidia-docker run  -p 8808:8808 -it detie:latest bash
```

Once this docker image starts, we're ready for work.

## Taking a minute to read the configs

This project uses [hydra](https://hydra.cc/) library for storing and changing the systems' metadata. The entry point 
to the arguments list that will be used upon running the scripts is the `config/config.yaml` file.

```yaml
defaults:
  - model: detie-cut
  - opt: adam
  - benchmark: carb
```

`model` leads to `config/model/...` subdirectory; please see [detie-cut.yaml](/config/model/detie-cut.yaml) 
for the parameters description.

`opt/adam.yaml` and `benchmark/carb.yaml` are the examples of configurations for the optimizer and the benchmark used.

If you want to change some of the parameters (e.g. `max_epochs`), not modifying the *.yaml files, just run e.g.

```bash
PYTHONPATH=. python some_..._script.py model.max_epochs=2
```

## Training

```
PYTHONPATH=. python3 modules/model/train.py
```

## Inference time

```
PYTHONPATH=. python3 modules/model/test.py model.best_version=243
```

This yields time in seconds when running inference against 
`modules/model/evaluation/oie-benchmark-stanovsky/raw_sentences/all.txt`
using batch size equal to 32.

Should be 708.6 sentences/sec. on NVIDIA Tesla V100 GPU.

## Evaluation

### English sentences

To apply the model to CaRB sentences, run 
```
cd modules/model/evaluation/carb-openie6/
PYTHONPATH=<repo root> python3 detie_predict.py
head -5 systems_output/detie243_output.txt
```

This will save the predictions into the `modules/model/evaluation/carb-openie6/systems_output/` directory. The same
should be done with `modules/model/evaluation/carb-openie6/detie_conj_predictions.py`.

To reproduce the DetIE numbers from the Table 3 in the paper, run

```bash
cd modules/model/evaluation/carb-openie6/
./eval.sh
```

* `detie243` is a codename for DetIE_{LSOIE}
* `detie243conj` is a codename for DetIE_{LSOIE} + IGL-CA
* `detie263` is a codename for DetIE_{IMoJIE}
* `detie263conj` is a codename for DetIE_{IMoJIE} + IGL-CA


## Synthetic data

To generate sentences using Wikidata's triplets, one can run the scripts

```
PYTHONPATH=. python3 modules/scripts/data/generate_sentences_from_triplets.py  wikidata.lang=<lang> 
PYTHONPATH=. python3 modules/scripts/data/download_wikidata_triplets.py  wikidata.lang=<lang>
```
 
 # Cite
 Please cite the original paper if you use this code.
 
 ```bibtex
@inproceedings{Vasilkovsky2022detie,
    author    = {Michael Vasilkovsky, Anton Alekseev, Valentin Malykh, Ilya Shenbin, Elena Tutubalina, 
                Dmitriy Salikhov, Mikhail Stepnov, Andrei Chertok and Sergey Nikolenko},
    title     = {{DetIE: Multilingual Open Information Extraction Inspired by Object Detection}},
    booktitle = {
        {Proceedings of the 36th {AAAI} Conference on Artificial Intelligence}
    },
    year      = {2022}
  }
```
 
 
 # Contact
 
Michael Vasilkovsky  `waytobehigh (at) gmail (dot) com` 
