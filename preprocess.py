from collections import Counter

import torch
from torch.autograd import Variable
import numpy as np
from nltk import word_tokenize

import config


def seq_padding(X, padding=0):
    """
    add padding to a batch data
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class PrepareData:
    def __init__(self, train_file, dev_file):
        # 01. Read the data and tokenize
        self.train_en, self.train_cn = self.load_data(train_file)
        self.dev_en, self.dev_cn = self.load_data(dev_file)

        # 02. build dictionary: English and Chinese
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(
            self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(
            self.train_cn)

        # 03. word to id by dictionary
        self.train_en, self.train_cn = self.wordToID(
            self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(
            self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        # 04. batch + padding + mask
        self.train_data = self.splitBatch(
            self.train_en, self.train_cn, config.BATCH_SIZE)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, config.BATCH_SIZE)

    def load_data(self, path):
        """
        Read English and Chinese Data
        tokenize the sentence and add start/end marks(Begin of Sentence; End of Sentence)
        en = [['BOS', 'i', 'love', 'you', 'EOS'],
              ['BOS', 'me', 'too', 'EOS'], ...]
        cn = [['BOS', '我', '爱', '你', 'EOS'],
              ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        en = []
        cn = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])
                cn.append(
                    ["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"])
        return en, cn

    def build_dict(self, sentences, max_words=50000):
        """
        sentences: list of word list
        build dictionary as {key(word): value(id)}
        """
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = config.UNK
        word_dict['PAD'] = config.PAD
        # inverted index: {key(id): value(word)}
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):
        """
        convert input/output word lists to id lists.
        Use input word list length to sort, reduce padding.
        """
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        def len_argsort(seq):
            """
            get sorted index w.r.t length.
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort:  # update index
            sorted_index = len_argsort(out_en_ids)  # English
            out_en_ids = [out_en_ids[id] for id in sorted_index]
            out_cn_ids = [out_cn_ids[id] for id in sorted_index]
        return out_en_ids, out_cn_ids

    def splitBatch(self, en, cn, batch_size, shuffle=True):
        """
        get data into batches
        """
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # paddings: batch, batch_size, batch_MaxLength
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))
            #!!! 'Batch' Class is called here but defined in later section.
        return batches


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        # convert words id to long format.
        src = torch.from_numpy(src).to(config.DEVICE).long()
        trg = torch.from_numpy(trg).to(config.DEVICE).long()
        self.src = src
        # get the padding postion binary mask
        # change the matrix shape to  1×seq.length
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            # decoder input from target
            self.trg = trg[:, :-1]
            # decoder target from trg
            self.trg_y = trg[:, 1:]
            # add attention mask to decoder input
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # check decoder output padding number
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask  # subsequent_mask is defined in 'decoder' section.


def subsequent_mask(size):
    """
    这里使用了上三角函数，用来生成在 DecoderLayer 中用到的 Mask Attention 的 Mask
    这个就是 target_mask
    :param size:
    :return:
    """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0