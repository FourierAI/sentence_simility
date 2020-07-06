#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: loaddata.py
# @time: 2020-07-06 09:55
# @desc:

import pandas as pd
import numpy as np
import csv
import torch


def load_data_csv(data_set, type):
    if data_set == "sts":
        if type == "train":
            data_frame = pd.read_csv('../dataset/stsbenchmark/sts-train.csv', delimiter='\t', error_bad_lines=False,
                                     quoting=csv.QUOTE_NONE)
        elif type == "test":
            data_frame = pd.read_csv('../dataset/stsbenchmark/sts-test.csv', delimiter='\t', error_bad_lines=False,
                                     quoting=csv.QUOTE_NONE)
        elif type == 'dev':
            data_frame = pd.read_csv('../dataset/stsbenchmark/sts-dev.csv', delimiter='\t', error_bad_lines=False,
                                     quoting=csv.QUOTE_NONE)
        else:
            data_frame = None

    return data_frame.dropna(how='any')


if __name__ == "__main__":
    train_df = load_data_csv(data_set='sts', type='train')

    train_score = train_df.iloc[:, [4]]
    train_sents_left = train_df.iloc[:, [5]]
    train_sents_right = train_df.iloc[:, [6]]

    print(train_score)

    print(train_sents_left)

    print(train_sents_right)
