#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: embeddingword.py
# @time: 2020-07-06 11:54
# @desc:

import dataprocess.loaddata as ld
from gensim.models import Word2Vec
from config.embeddingconfig import EmbeddingConfig
import nltk
from nltk.tokenize import word_tokenize


# pre_train word2vec
def pre_train_word2vec():

    all_sents = []
    all_sents.extend(train_sents_left)
    all_sents.extend(train_sents_right)
    all_sents.extend(test_sents_left)
    all_sents.extend(test_sents_right)
    all_sents.extend(dev_sents_left)
    all_sents.extend(dev_sents_right)
    sents = [word_tokenize(sent[0].lower()) for sent in all_sents]
    model = Word2Vec(sents, sg=config.Word2Vec_model, size=config.dimension,
                     window=config.Word2Vec_WindowSize, min_count=config.Word2Vec_minicount,
                     workers=config.Word2Vec_Worker)
    model.save('Word2Vec.model')
    return model


if __name__ == "__main__":
    config = EmbeddingConfig()

    train_df = ld.load_data_csv(data_set='sts', type='train')
    test_df = ld.load_data_csv(data_set='sts', type='test')
    dev_df = ld.load_data_csv(data_set='sts', type='dev')

    train_sents_left = train_df.iloc[:, [5]].values.tolist()
    train_sents_right = train_df.iloc[:, [6]].values.tolist()

    test_sents_left = test_df.iloc[:, [5]].values.tolist()
    test_sents_right = test_df.iloc[:, [6]].values.tolist()

    dev_sents_left = dev_df.iloc[:, [5]].values.tolist()
    dev_sents_right = dev_df.iloc[:, [6]].values.tolist()

    if config.pre_train:
        word2vec_model = pre_train_word2vec()
        # for

    else:
        pass
