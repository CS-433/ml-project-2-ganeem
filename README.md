[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)



# Tweet Sentiment analysis

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
</p>

## About

This project aims to do tweets sentiment classification. We use [epfl-ml-text-classification](https://www.aicrowd.com/challenges/epfl-ml-text-classification) dataset.

## Installation

Follow these steps:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How To Use

Download the dataset from AIcrowd and put them inside a folder called `data`, at the root of the project. (/data/train_pos.txt for example)

You can then launch the jupyterlab environement to run the notebook with:
```bash
jupyter lab
```
Then run the cells inside the notebook `finetuning.ipynb`. 

It contains the code that was used to fine-tune the model.
