import time

import torch
from torch import nn

import config
from models.model import make_model
from preprocess import PrepareData


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
