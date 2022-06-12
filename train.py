import time

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable

import config
from models.model import make_model
from preprocess import PrepareData, subsequent_mask


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)

        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.)

        self.true_dist = true_dist

        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm.float()


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def run_epoch(data, model, loss_compute, epoch):
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    for i, batch in enumerate(data):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch {:d} Batch: {:d} Loss: {:.4f} Tokens per Sec: {:.2f}s".format(
                epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed / 1000.)))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


def train(data, model, criterion, optimizer):
    """
    Train and Save the model.
    """
    # init loss as a large value
    best_dev_loss = 1e5

    for epoch in range(config.EPOCHS):
        # Train model
        model.train()
        run_epoch(data.train_data, model, SimpleLossCompute(
            model.generator, criterion, optimizer), epoch)
        model.eval()

        # validate model on dev dataset
        print('>>>>> Evaluate')
        dev_loss = run_epoch(data.dev_data, model, SimpleLossCompute(
            model.generator, criterion, None), epoch)
        print('<<<<< Evaluate loss: {:.2f}'.format(dev_loss))

        # save the model with best-dev-loss
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            # SAVE_FILE = 'save/model.pt'
            torch.save(model.state_dict(), config.SAVE_FILE)

        print(f">>>>> current best loss: {best_dev_loss}")


if __name__ == '__main__':

    # Step 1: Data Preprocessing
    data = PrepareData(config.TRAIN_FILE, config.DEV_FILE)
    src_vocab = len(data.en_word_dict)
    tgt_vocab = len(data.cn_word_dict)
    print(f"src_vocab {src_vocab}")
    print(f"tgt_vocab {tgt_vocab}")

    # Step 2: Init model
    model = make_model(
        src_vocab,
        tgt_vocab,
        config.LAYERS,
        config.D_MODEL,
        config.D_FF,
        config.H_NUM,
        config.DROPOUT
    )

    # Step 3: Training model
    print(">>>>>>> start train")
    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab, padding_idx=0, smoothing=0.0)
    optimizer = NoamOpt(config.D_MODEL, 1, 2000, torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    model.to(config.DEVICE)

    train(data, model, criterion, optimizer)
    print(f"<<<<<<< finished train, cost {time.time() - train_start:.4f} seconds")


# %% 6. Prediction with English-to-Chinese Translator

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Translate src with model
    """
    # decode the src
    memory = model.encode(src, src_mask)
    # init 1×1 tensor as prediction，fill in ('BOS')id, type: (LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    #  run 遍历输出的长度下标
    for i in range(max_len-1):
        # decode one by one
        out = model.decode(memory,
                           src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        #  out to log_softmax
        prob = model.generator(out[:, -1])
        #  get the max-prob id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        #  concatnate with early predictions
        ys = torch.cat([ys, torch.ones(1, 1).type_as(
            src.data).fill_(next_word)], dim=1)
    return ys


# English to Chinese Translations

def evaluate(data, model):
    """
    Make prediction with trained model, and print results.
    """
    with torch.no_grad():
        #  pick some random sentences from dev data.
        for i in np.random.randint(len(data.dev_en), size=10):
            # Print English sentence
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)

            # Print Target Chinese sentence
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("".join(cn_sent))

            # conver English to tensor
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(config.DEVICE)
            src = src.unsqueeze(0)
            # set attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # apply model to decode, make prediction
            out = greedy_decode(
                model, src, src_mask, max_len=config.MAX_LENGTH, start_symbol=data.cn_word_dict["BOS"])
            # save all in the translation list
            translation = []
            # convert id to Chinese, skip 'BOS' 0.
            # 遍历翻译输出字符的下标（注意：跳过开始符"BOS"的索引 0）
            for j in range(1, out.size(1)):
                sym = data.cn_index_dict[out[0, j].item()]
                if sym != 'EOS':
                    translation.append(sym)
                else:
                    break
            print("translation: {}".format(" ".join(translation)))


# **English to Chinese Translator**

# Predition
model.load_state_dict(torch.load(config.SAVE_FILE))
print(">>>>>>> start evaluate")
evaluate_start = time.time()
evaluate(data, model)
print(
    f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")


# Here is my pretrained model with CUDA support
if config.DEVICE == "cuda":
    SAVE_FILE_EXTRA = 'save/models/large_model.pt'
    model.load_state_dict(torch.load(SAVE_FILE_EXTRA))
    print(">>>>>>> start evaluate")
    evaluate_start = time.time()
    evaluate(data, model)
    print(
        f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")
