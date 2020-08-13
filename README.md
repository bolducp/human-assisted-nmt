# human-assisted-NMT
UW masters thesis in Computational Linguistics

## Overview
In this thesis, we will build a human-assisted NMT system that aims to maximize the quality of translation, while reducing the total amount of human translator effort involved. At a high level, we construct a system in which we endeavor to use the NMT translation directly when it is of sufficiently high quality, and ask for human feedback to improve the translation only when it is poor. 

The system has two modeling components. The first is a pre-existing black-box NMT model, with which we can obtain a predicted translation for any input. The second is a feedback-requester model, which uses the output of the NMT system, the source sentence, (and possibly additional features of the translation such as the output of a grammar checker) to predict for each sentence whether human translator feedback should be solicited to improve the NMT translation. In addition, the system will have an interface through which requests for human feedback will be made and used to correct the translation. The interface will also display the translated document upon completion for the human translator to further post-edit if desired. All human feedback provided at any stage will be used to improve the feedback-requester model.

## Project Structure

```
human-assisted-NMT
│ README.md, MIT license, requirements.txt, setup.py, etc
│     
│
└───hnmt
│   └───nmt
│   │    │ This directory contains all of the setup and processing necessary for our chosen
│   │    │ "black box" pretrained NMT model.  
│   │    │ Instructions provided below for how to download and tokenize the  NMT system output
│   │    │ to use as input into the feedback-requestor for training and inference.
│   │  
│   │
│   │       
│   │   
│   └───feedback-requester
│         │ Here lives all of the code for the feedback-requester model that uses the NMT output
│         │ to determine whether to prompt a translator for feedback on a given sentence.
│   
│       
│      
│        
└───tests
    │   all of the unit tests live here 
```

## Pretrained NMT model setup and output generation

For this initial proof-of-concept for our system, we are using a pretrained 
Japanese-English NMT model [trained on the JParaCrawl corpus](https://github.com/MorinoseiMorizo/jparacrawl-finetune).

### Requirements for obtaining outputs from the JParaCrawl pretrained model

These instructions assume that you have two files to use as the parellel sentences
saved in some subdirectory of the `nmt` folder (e.g. "/corpus/spm/kyoto-train.ja"
and "/corpus/kftt-data-1.0/data/tok/kyoto-train.en" if using the KFTT corpus-- an
example for how to download and preprocess this data can be found on the JParaCrawl
github).

1. Make sure that you have all of the dependencies installed. NOTE: per the JParaCrawl github instructions, make sure to use the same version of the fairseq library that they used when pre-training the model:
```
$ cd fairseq
$ git checkout c81fed46ac7868c6d80206ff71c6f6cfe93aee22
```
2. Download the pretrained model from the JParaCrawl [website](http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/). There are three sizes available (small, base, and large). Make sure that the `checkpoint_file` parameter used to instaniate the `pretrained_nmt_model` in `nmt/main.py` correctly references the size of the model you downloaded.
3. Create a sub-folder in `nmt` called `pretrained_model_jaen` for the pretrained model (should include `dict.en.txt`, `dict.ja.txt`, `LICENSE`, and the model file, e.g. `small.pretrain.pt`).
4. Ensure that the `tok_jpn_sents_path` and `tok_en_sents_path` variables in `nmt/main.py` point to your local data paths.
5. Run `python3 main.py`, which will save the pickled output to a file called `nmt_out.p` in the `nmt` directory.

### Using output from a different NMT system
To be used as input into the feedback-requester model, NMT output should have the following form List[Tuple[Tuple[str, torch.Tensor], str, str]], where each item in the list represents the NMT system output for a given sentence and the true reference translation. Specifically, each item should be:

((str: predicted nmt output, torch.Tensor: probabilty score for each word piece), str: input source text, str: gold translation target)

For example, here is a single element in the output list:
```
(('Japanese ink-wash painting was completely changed.',
  tensor([-2.2287, -0.1886, -1.6461, -0.5150, -0.6565, -1.1917, -0.8082, -0.9688,
          -0.1424, -0.0960])),
 '▁日本の 水 墨 画 を 一 変 させた 。',
 'He revolutionized the Japanese ink painting .')
```



## Installation and running the feedback-requestor

## Experiments