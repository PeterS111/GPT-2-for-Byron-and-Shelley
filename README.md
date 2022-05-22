# GPT-2-for-Byron-and-Shelley
version 04.05.2022

This repository contains data and code for the paper:

"Training GPT-2 to represent two Romantic-era authors: challenges, evaluations and pitfalls" 

by Piotr Sawicki, Marek Grzes, Anna Jordanous, Dan Brown, Max Peeperkorn

ICCC 2022

## Contents of the repository

**"datasets"** folder contains the original datasets of collected works by Byron and Shelley, whihc were used to fine-tune our models. We provide both the original datasets (as downloaded from Gutenberg.org), and their pre-processed versions, which were used in the experiments. 

**"datasets_for_experiments_1_and_2"** contains the complete datasets used for Experiments 1 and 2, for descriptions please refer to our paper

**"datasets_of_plagiarized-samples"** contains the samples which has one or more lines of text that was plagiarized from the original datasets. This experiment was now included in the paper, we only briefly mentioned the results. In this experiments we generated we fine-tuned GPT-2 small and medium models for 100K steps, generating 1K samples at each 10K steps interval. After that, we run a python script that compared each line in each sample to each line in the original dataset. We only considered the lines that has length greater than 3, and contain less then 70% of capital letters (to exclude chapter names). The samples for each checkpoint were ordered by the number of repeated lines, and reprinted in .html format, where the repeated lines are printed in red. For example:

Sample "0_48_08809.html" in the folder "output_Byron_Regular_Medium" has position "0" (first), as it contains the highest number of duplicate lines, number "48" is the number of repeated lines, "08809" indicates that it was generated at 80K steps checkpoint with a seed "809",
Samples "111_20_06267.html" has position 111, with 20 repeated lines, generated at 60K steps checkpoint, with seed "267", etc.
