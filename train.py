import argparse
import shutil
import os
import types

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for missing PyYAML
    from minimal_yaml import safe_load
    yaml = types.SimpleNamespace(safe_load=safe_load)

try:
    from torch.optim.lr_scheduler import OneCycleLR
    from torch.utils.data import DataLoader
    from torch.nn import CrossEntropyLoss
    from torch import load, ones, save
    from torch.optim import AdamW
    MISSING_TORCH = False
except ModuleNotFoundError:  # pragma: no cover - allow running without torch
    OneCycleLR = DataLoader = CrossEntropyLoss = load = ones = save = AdamW = None
    MISSING_TORCH = True

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - fallback if tensorboardX missing
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception:
        class SummaryWriter:  # type: ignore
            def __init__(self, *a, **k):
                pass

            def add_scalar(self, *a, **k):
                pass


"""Training script for the GPT model.

This module provides utilities for running training epochs and saving
checkpoints for later use. It exposes a ``main`` function that loads
configuration files and manages the training loop.
"""


def save_checkpoint(path, model, opt, sch, epoch):
    """Save a checkpoint of the current training state.

    Args:
        path (str): Directory to store the checkpoint.
        model (torch.nn.Module): Model to save.
        opt (Optimizer): Optimizer whose state to persist.
        sch (Scheduler): Scheduler whose state to persist.
        epoch (int): Current epoch number.
    """

    filepath = f'{path}/epoch_{epoch}'
    if os.path.exists(filepath):
        shutil.rmtree(filepath)

    os.makedirs(filepath)
    save(model.state_dict(), f'{filepath}/model.pth')
    save(opt.state_dict(), f'{filepath}/opt.pth')
    save(sch.state_dict(), f'{filepath}/sch.pth')


def publish_metrics(logger, train_metrics, dev_metrics, epoch):
    """Publish training and development metrics to TensorBoard.

    Args:
        logger (SummaryWriter): Logger instance used to write metrics.
        train_metrics (dict): Dictionary of metrics from the training set.
        dev_metrics (dict): Dictionary of metrics from the dev set.
        epoch (int): Current epoch number.
    """

    for key in train_metrics:
        logger.add_scalar(f'train_{key}', train_metrics[key], epoch)

    for key in dev_metrics:
        logger.add_scalar(f'dev_{key}', dev_metrics[key], epoch)


def main():
    """Entry point for training the GPT model."""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--confpath', type=str, required=True)
    parser.add_argument('-ch', '--checkpoint', type=str, default=None)
    args = parser.parse_args()
    confpath = args.confpath
    checkpoint = args.checkpoint

    if MISSING_TORCH:  # pragma: no cover - informative exit if torch missing
        print('PyTorch is required to run training. Please install torch.')
        return

    from model.dataset import TokenIDDataset, TokenIDSubset
    from model.trainer import Trainer
    from model.model import GPT

    confs = yaml.safe_load(open(confpath))

    train_data = TokenIDDataset(**confs['train_data'])
    dev_data = TokenIDDataset(**confs['dev_data'])

    model = GPT(**confs['model'])
    opt = AdamW(model.get_parameters(), **confs['opt'])
    sch = OneCycleLR(opt, **confs['sch'])
    crit = CrossEntropyLoss(ignore_index=confs['unk'])
    trainer = Trainer(model, crit, opt, sch, **confs['trainer'])
    logger = SummaryWriter(**confs['logger'])

    start_epoch = 0
    if checkpoint is not None:
        model.load_state_dict(load(f'{checkpoint}/model.pth'))
        opt.load_state_dict(load(f'{checkpoint}/opt.pth'))
        sch.load_state_dict(load(f'{checkpoint}/sch.pth'))
        start_epoch = int(checkpoint.split('epoch_')[-1].strip('/'))

    for epoch in range(start_epoch, confs['epochs']):

        print(f'\n\nEpoch {epoch+1}')
        train = TokenIDSubset(train_data, **confs['train_subset'])
        dev = TokenIDSubset(dev_data, **confs['dev_subset'])

        collate = TokenIDDataset.collate
        tloader = DataLoader(
            collate_fn=collate,
            **confs['loader'],
            dataset=train,
        )
        dloader = DataLoader(
            collate_fn=collate,
            **confs['loader'],
            dataset=dev,
        )

        train_metrics = trainer.run_epoch(tloader)
        dev_metrics = trainer.run_epoch(dloader, train_mode=False)
        publish_metrics(logger, train_metrics, dev_metrics, epoch+1)
        save_checkpoint(confs['checkpoint'], model, opt, sch, epoch+1)


if __name__ == '__main__':
    main()
