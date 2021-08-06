This is the repository for the
paper [Reading StackOverflow Encourages Cheating: Adding Question TextImproves Extractive Code Generation](https://arxiv.org/abs/2106.04447)
.

![Our Approach](https://github.com/gabeorlanski/stackoverflow-encourages-cheating/blob/main/data/approach_figure.PNG)

![Labeled Example](https://github.com/gabeorlanski/stackoverflow-encourages-cheating/blob/main/data/labeled_example.PNG)

## Acknowledgements

We would like to thank Frank F. Xu and Pengcheng Yin for their helpful discussions and for sharing
their code. Some code has come from the [TranX](https://github.com/pcyin/tranx)
and [External Knowledge Codegen](https://github.com/neulab/external-knowledge-codegen) repositories.

We would also like to thank the work that inspired this one:

[TRANX: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation](https://www.aclweb.org/anthology/D18-2002/)
by Pengcheng Yin and Graham Neubig

[Incorporating External Knowledge through Pre-training for Natural Language to Code Generation](https://www.aclweb.org/anthology/2020.acl-main.538/)
by Frank F. Xu, Zhengbao Jiang, Pengcheng Yin, Bogdan Vasilescu, and Graham Neubig


## TL;DR For Replication

Run the Google colab
found [Notebook Link](https://github.com/gabeorlanski/stackoverflow-encourages-cheating/blob/main/BART_CG_Experiments.ipynb) [![Open Replication In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabeorlanski/stackoverflow-encourages-cheating/blob/main/BART_CG_Experiments.ipynb)
for our best performing model.

We also provide all of the generated samples from our test with the
inputs [here](https://github.com/gabeorlanski/stackoverflow-encourages-cheating/blob/main/data/generated.txt)
.

Note: It will take 1-2 (Maybe 3) hours to train and run on Google Colab

## For working outside of colab

You need Python to use Python 3.8. I would recommend using a virtual environment.

1. Install the requirements
   from [`requirements.txt`](https://github.com/gabeorlanski/stackoverflow-encourages-cheating/blob/main/requirements.txt)

```shell script
pip install -r requirements.txt
```

2. To run the model, run
   the [`experiment.py`](https://github.com/gabeorlanski/stackoverflow-encourages-cheating/blob/main/experiment.py)
   script. You can use `python experiment.py -h` or the documentation in the file to understand the
   different options. But to use our best model, run

```shell script
python experiment.py best "facebook/bart-base" bartBase -combine-mined
```

3. Then in the `scratch` directory, you will find the results in a json file.

## The data

### Prepared Dataset:

[Here](https://www.dropbox.com/s/xv3zcutli07w37w/base_dataset.zip?dl=0) is our dataset that we used.

[This dataset](https://www.dropbox.com/s/glioprd0aly4381/cleaned_so_dataset.rar?dl=0) is the _cleaned_ data using the process we describe further down. **NOTE** For the time being this only includes 10,000 mined examples. It will be updated to include all cleaned mined examples.

You can find a sample schema for this
data [here](https://github.com/gabeorlanski/stackoverflow-encourages-cheating/blob/main/data/base_dataset_sample.json)
.

For the `body` key, there are unclosed html tags in the text. *Eventually* these will be taken out.
But for now, the easy but bad solution is to use the regex `<\w+>`. The good solution is to use
the [html tags file](https://github.com/gabeorlanski/stackoverflow-encourages-cheating/blob/main/data/html_tags.txt)
to remove them. Note, you must surround the tag text with `< >`.

### Parsed StackOverflow Data:

[Link to the parsed StackOverflow Questions](https://www.dropbox.com/s/glioprd0aly4381/cleaned_so_dataset.rar?dl=0)

For actually working with this data:

1. The JSON file has the structure:

```json
{
    "question_id": {
        "question_id": "str",
        "tags": "List[str]",
        "title": "str",
        "accepted_answer_id": "int or null",
        "score": "int",
        "body": "str",
        "code_slots": "Ignore this, it is useless",
        "answers": {
            "answer_id": {
                "score": "int",
                "body": "str",
                "code_slots": "Ignore"
            }
        }
    }
}
``` 

2. For the `body` key, there are unclosed html tags in the text. *Eventually* these will be taken
   out. But for now, the easy but bad solution is to use the regex `<\w+>`. The good solution is to
   use
   the [html tags file](https://github.com/gabeorlanski/stackoverflow-encourages-cheating/blob/main/data/html_tags.txt)
   to remove them. Note, you must surround the tag text with `< >`.

3. Finally, you must match the question ids from CoNaLa to the SO data.

## References

If you use this dataset you MUST cite the [original CoNaLa paper](https://conala-corpus.github.io/) as well:

```
@misc{orlanski2021reading,
      title={Reading StackOverflow Encourages Cheating: Adding Question Text Improves Extractive Code Generation}, 
      author={Gabriel Orlanski and Alex Gittens},
      year={2021},
      eprint={2106.04447},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
@inproceedings{yin2018mining,
  author = {Yin, Pengcheng and Deng, Bowen and Chen, Edgar and Vasilescu, Bogdan and Neubig, Graham},
  title = {Learning to Mine Aligned Code and Natural Language Pairs from Stack Overflow},
  booktitle = {International Conference on Mining Software Repositories},
  series = {MSR},
  pages = {476--486},
  year = {2018},
  publisher = {ACM},
  doi = {https://doi.org/10.1145/3196398.3196408},
}
```
