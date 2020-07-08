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
from config.global_config import GlobalConfig
from nltk.tokenize import word_tokenize
import torch
import pickle


# pre_train word2vec
def pre_train_word2vec(sents, embedding_config):
    model = Word2Vec(sents, sg=embedding_config.Word2Vec_model, size=embedding_config.dimension,
                     window=embedding_config.Word2Vec_WindowSize, min_count=embedding_config.Word2Vec_minicount,
                     workers=embedding_config.Word2Vec_Worker)
    model.save('Word2Vec.model')
    return model


def sent2tensor(sentences, word2vec_or_embedding, word2vec_model=None, word_embedding=None, word_dict=None):
    sents_embedding = []
    sents = [word_tokenize(sent[0].lower()) for sent in sentences]
    for sent in sents:
        sent_embedding = []
        for word in sent:
            if word2vec_or_embedding == 'word2vec':
                sent_embedding.append(word2vec_model.wv.__getitem__(word).tolist())
            else:
                sent_embedding.append(word_embedding(torch.LongTensor([word_dict[word]])).tolist())
        sent_embedding_tensor = torch.FloatTensor(sent_embedding)
        sents_embedding.append(sent_embedding_tensor)
    return sent_embedding


def sent2tensors(data_set):
    global_config = GlobalConfig()
    embedding_config = EmbeddingConfig()
    train_sents_embedding_left = []
    train_sents_embedding_right = []
    if global_config.train_or_test == "train":

        train_sents_left, train_sents_right = load_sents_pair()
        whole_sents = ld.load_all_sents(data_set)

        if embedding_config.pre_train:
            word2vec_model = pre_train_word2vec(whole_sents, embedding_config)
            train_sents_embedding_left = sent2tensor(sentences=train_sents_left, word2vec_or_embedding='word2vec',
                                                     word2vec_model=word2vec_model)
            train_sents_embedding_right = sent2tensor(sentences=train_sents_right, word2vec_or_embedding='word2vec',
                                                      word2vec_model=word2vec_model)
        else:
            word_set, word_dict = get_wordset_and_worddict(whole_sents)
            word_capability = len(word_set)
            word_embedding = torch.nn.Embedding(word_capability, embedding_config.dimension)
            with open('word_embedding.model', 'wb') as file:
                pickle.dump(word_embedding, file)

            train_sents_embedding_left = sent2tensor(sentences=train_sents_left, word2vec_or_embedding='embedding',
                                                     word_embedding=word_embedding, word_dict=word_dict)
            train_sents_embedding_right = sent2tensor(sentences=train_sents_right, word2vec_or_embedding='embedding',
                                                      word_embedding=word_embedding, word_dict=word_dict)
    elif global_config.train_or_test == "test":



    return train_sents_embedding_left, train_sents_embedding_right


def get_wordset_and_worddict(whole_sents):
    word_set = set()
    word_dict = {}
    index = 0
    for sent in whole_sents:
        for word in sent:
            if word not in word_set:
                word_dict[word] = index
                word_set.add(word)
                index += 1
    return word_set, word_dict


def load_all_sents(train_sents_left, train_sents_right):
    global whole_sents
    test_df = ld.load_data_csv(data_set='sts', type='test')
    dev_df = ld.load_data_csv(data_set='sts', type='dev')
    test_sents_left = test_df.iloc[:, [5]].values.tolist()
    test_sents_right = test_df.iloc[:, [6]].values.tolist()
    dev_sents_left = dev_df.iloc[:, [5]].values.tolist()
    dev_sents_right = dev_df.iloc[:, [6]].values.tolist()
    all_sents = []
    all_sents.extend(train_sents_left)
    all_sents.extend(train_sents_right)
    all_sents.extend(test_sents_left)
    all_sents.extend(test_sents_right)
    all_sents.extend(dev_sents_left)
    all_sents.extend(dev_sents_right)
    whole_sents = [word_tokenize(sent[0].lower()) for sent in all_sents]


def load_sents_pair():
    train_df = ld.load_data_csv(data_set='sts', type='train')

    train_sents_left = train_df.iloc[:, [5]].values.tolist()
    train_sents_right = train_df.iloc[:, [6]].values.tolist()

    return train_sents_left, train_sents_right


if __name__ == "__main__":
    train_sents_embedding_left, train_sents_embedding_right = sent2tensors('sts')
