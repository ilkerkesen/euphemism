import os
import os.path as osp

import pytorch_lightning as pl


def preprocess_path(path):
    return osp.abspath(osp.expanduser(path))


def create_callbacks(config, log_dir):
    checkpoints_path = osp.join(log_dir, 'checkpoints')
    config['checkpoint']['dirpath'] = checkpoints_path
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**config['checkpoint'])
    last_ckpt = osp.join(checkpoints_path, 'last.ckpt')
    last_ckpt = last_ckpt if osp.isfile(last_ckpt) else None
    ckpt_path = config['trainer']['resume_from_checkpoint']

    if last_ckpt is not None and ckpt_path is not None:
        raise Exception('resume checkpoint passed (last.ckpt exists already)')

    ckpt_path = last_ckpt if ckpt_path is None else ckpt_path
    if ckpt_path is not None and not osp.isfile(ckpt_path):
        raise Exception('ckpt does not exist at {}'.format(ckpt_path))

    return [checkpoint_callback], ckpt_path


def create_logger(config):
    assert config['logger'].get('version') is not None
    if config['logger']['version'] == 'debug':
        return None
    config['logger']['save_dir'] = osp.abspath(
        osp.expanduser(config['logger']['save_dir']))
    if config['logger']['name'] is None:
        model_config = config.get('model', {})
        architecture = model_config.get('name', 'SomeModel')
        config['logger']['name'] = f'{architecture}'
    logger = pl.loggers.TensorBoardLogger(**config['logger'])
    return logger


def process_config(config):
    model_config = config.get('model', {})
    text_encoder = model_config.get('text_encoder', 'microsoft/deberta-base')
    data_config = config.get('data', {})
    data_config['tokenizer'] = text_encoder
    config['data'] = data_config
    return config


def write_results(file_path, predictions):
    file_path = preprocess_path(file_path)
    with open(file_path, 'w') as f:
        for x in predictions:
            indexes, probs = x['indexes'], x['predictions']
            for i in range(len(indexes)):
                index, prob = int(indexes[i]), float(probs[i])
                f.write(f'{index},{prob:0.4f}\n')