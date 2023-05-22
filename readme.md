# Project Summary

This project was done as part of the course CS6120-Natural Language Processing.

Here, we investigated the efficacy of different RNN-based language modeling architectures on the task of music lyrics classification and generation. Our custom dataset contains 80k song lyrics of the genres *Rock, Pop, Rap,* and *Country*. The dataset is available [**here**](https://www.kaggle.com/datasets/novanglus/music-lyrics-by-genre).

## Classification
We evaluate the classification performance of three main architectures: Vanilla RNN, GRU, and a number of LSTM variations (including multilayer and bidirectionality). Our best-performing classifier was the GRU and achieved a macro-F1 score of 0.71 and validation accuracy of 72%. 

The code for the analysis is in E1 folder and is called RNN_Analysis.ipynb. The instructions on how to run the doc the code are at the top of the notebook.

## Generation
We have trained the RNN, LSTM, and GRU architectures on the task of lyrics generation and found that LSTM consistently produced lyrics of higher quality. We also experiment with how the initial cell-state and the hidden-state can be used to nudge the LSTM-based model to generate lyrics in a specific genre.

## Dataset Creation

Instructions to run the code for creating the dataset:

1.) You need to create an account on Genius.com and get an API key in order to run the code.

2.) You need to download the artists dataset from here: https://www.kaggle.com/datasets/pieca111/music-artists-popularity?resource=download
