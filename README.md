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
│   │    │ This directory contains all of the setup and processing necessary for
│   │    │ our chosen "black box" pretrained NMT model.  
│   │    │ Instructions provided below for how to download and tokenize the 
│   │    │ parallel corpora and get the NMT system output to use as input into 
│   │    │ the feedback-requestor for training and inference.
│   │
│   │       
│   │   
│   └───feedback-requester
│         │ Here lives all of the code for the feedback-requester model that
│         │ uses the NMT output to determine whether to prompt a translator for 
│         │ feedback on a given sentence.
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

## Installation and running the feedback-requestor

## Experiments