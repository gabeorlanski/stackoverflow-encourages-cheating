This is still very much a WIP.

## Acknowledgements

This is the repository for the paper **ADD TITLE AND LINK**. Some code has come from the [TranX](https://github.com/pcyin/tranx) 
and [External Knowledge Codegen](https://github.com/neulab/external-knowledge-codegen) repositories.

## TL;DR For Replication

Run the Google colab found **ADD LINK** for our best performing model.

We also provide all of the generated samples from our test with the inputs: [Notebook Link](https://github.com/gabeorlanski/noisy-so-code-generation/blob/main/BART_CG_Experiments.ipynb) [![Open Replication In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/gabeorlanski/noisy-so-code-generation/blob/main/BART_CG_Experiments.ipynb)

Note: It will take 1-2 hours to train and run on Google Colab

## For working outside of colab

You need Python to use Python 3.8. I would recommend using a virtual environment.

1. Install the requirements from [`requirements.txt`](https://github.com/gabeorlanski/noisy-so-code-generation/blob/main/requirements.txt)
```shell script
pip install -r requirements.txt
```

2. To run the model, run the [`experiment.py`](https://github.com/gabeorlanski/noisy-so-code-generation/blob/main/experiment.py) script. You can use `python experiment.py -h` or 
the documentation in the file to understand the different options. But to use our best model, run 
```shell script
python experiment.py best "facebook/bart-base" bartBase -combine-mined
```

3. Then in the `scratch` directory, you will find the results in a json file. 

## The data

### Parsed StackOverflow Data:

[Link parsed StackOverflow Questions](https://www.dropbox.com/s/7l42q8foywuqu5y/parsed_so.zip?dl=0)

For actually working with this data:

1. The JSON file has the structure:
```json
{
    "question_id":{
        "question_id": "str",
        "tags": "List[str]",
        "title": "str",
        "accepted_answer_id": "int or null",
        "score": "int",
        "body": "str",
        "code_slots": "Ignore this, it is useless",
        "answers": {
               "answer_id":{
                    "score": "int",
                    "body": "str",
                    "code_slots": "Ignore"
                }    
        }
    }
}
``` 

2. For the `body` key, there are unclosed html tags in the text. *Eventually* these will be taken 
out. But for now, the easy but bad solution is to use the regex `<\w+>`. The good solution is to use 
the [html tags file](https://github.com/gabeorlanski/noisy-so-code-generation/blob/main/data/html_tags.txt) 
to remove them. Note, you must surround the tag text with `< >`.    

3. Finally, you must match the question ids from CoNaLa to the SO data.
