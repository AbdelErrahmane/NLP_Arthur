# Project Debrief: fine tune T5 (text-to-text transformer) on a custom sample corpus

A Shared Text-To-Text Framework
With T5, we propose reframing all NLP tasks into a unified text-to-text-format
where the input and output are always text strings, in contrast to BERT-style
models that can only output either a class label or a span of the input. Our
text-to-text framework allows us to use the same model, loss function, and 
hyperparameters on any NLP task, including machine translation, document summarization
, question answering, and classification tasks (e.g., sentiment analysis). 
We can even apply T5 to regression tasks by training it to predict the string 
representation of a number instead of the number itself.

Documentation [here](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)

## Requirement
Using python 3.9

## Installation

Yous should to install the requirement file requirements.txt, using the follow command:



```bash
pip install -r requirements.txt
```

we use this command to install:

```text
pandas == 1.2.4
sentencepiece
transformers
rich[jupyter]
torch==1.10.0
```

## Usage 


You should to change the parameters and the function in the main function.

```python
from utils import *

model_params={
    "MODEL":"t5-base",             # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE":8,          # training batch size
    "VALID_BATCH_SIZE":8,          # validation batch size
    "TRAIN_EPOCHS":3,              # number of training epochs
    "VAL_EPOCHS":1,                # number of validation epochs
    "LEARNING_RATE":1e-4,          # learning rate
    "MAX_SOURCE_TEXT_LENGTH":512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH":50,   # max length of target text
    "SEED": 42                     # set seed for reproducibility

}

my_df= pd.read_csv('test_corpus.csv')

my_df["s2t"] = "summarize: "+my_df["s2t"]

def main():
    print("Execution of th main function ...................")
    T5Trainer(dataframe=my_df[:500], source_text="s2t", target_text="original", model_params=model_params,
              output_dir="outputs")


if __name__ == "__main__":
    main()

```

## Conclusion
T5 can do all following tasks 
- translation 
- linguistic acceptability  
- sentence similarity 
- and document summarization

In our case , we use just sumarize 
  ```python
#summarization use the "summarize: " before the sentence
    my_df["s2t"] = "summarize: "+my_df["s2t"]  
```

to apply the  other process, we should do :
- translation 

```python
#Translate English to German , we can use "translate English to German: " before the sentence
    my_df["s2t"] = "translate English to German: "+my_df["s2t"]  
```

- linguistic acceptability

```python
#linguistic acceptability, we can use "cola sentence: " before the sentence
    my_df["s2t"] = "cola sentence: "+my_df["s2t"]  
```

- sentence similarity

```python
#sentence similarity  , we can use "stsb sentence1:<sentence1> sentence2:<sentence2> " before the sentence
    my_df["s2t"] = "stsb sentence1: "+my_df["s2t"] +"stsb sentence2:"+my_df["s1t"]
```