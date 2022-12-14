import os
import os.path as osp
import json
import re

from tqdm import tqdm
import spacy
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pytorch_lightning as pl


ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
DATA_ROOT = osp.join(ROOT, 'data')


class EuphemismDataset(Dataset):
    def __init__(self, items, split):
        super().__init__()
        self.items = items
        self.split = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class EuphemismDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root=DATA_ROOT,
        text_input='raw_sentence',
        use_definitions=False,
        use_images=False,
        use_hallucinations=False,
        batch_size=64,
        num_workers=0,
        tokenizer='microsoft/deberta-base',
        seed=42,
        val_percent=0.2,
    ):
        super().__init__()
        assert text_input in ('utterance', 'sentence', 'raw_sentence')
        self.root = osp.abspath(osp.expanduser(root))
        self.text_input = text_input
        self.use_definitions = use_definitions
        self.use_images = use_images
        self.use_hallucinations = use_hallucinations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.seed = seed
        self.torch_rng = torch.Generator().manual_seed(seed)
        self.val_percent = val_percent
        self.nlp = spacy.load("en_core_web_sm")

    def prepare_split(self, split='train', file_path=None):
        with open(osp.join(self.root, f'{split}.csv')) as f:
            raw = [line.strip() for line in f.readlines()][1:]
        data = []
        if split == 'train':
            pattern = r'(\d+)\,\"?(.+)\"?\,(0|1)'
        elif split == 'test':
            pattern = r'(\d+)\,\"?(.+)\"?'

        for line in tqdm(raw):
            groups = re.search(pattern, line)
            index = int(groups.group(1))
            utterance = groups.group(2)

            if split == 'train':
                label = int(groups.group(3))
            else:
                label = None

            doc = self.nlp(utterance)
            pre, post, found = [], [], False
            for sent in doc.sents:
                match = re.search(r"\<(.+)\>", sent.text)
                if match is None and not found:
                    pre.append(sent.text)
                elif match is None and found:
                    post.append(sent.text)
                else:
                    found = True
                    raw_sentence = sent.text
                    phrase = match.group(1)
                    lemma = ' '.join([t.lemma_ for t in self.nlp(phrase)])
                    sentence = raw_sentence.replace(f'<{phrase}>', phrase)

            data.append({
                'index': index,
                'utterance': utterance,
                'label': label,
                'pre': ' '.join(pre),
                'post': ' '.join(post),
                'raw_sentence': raw_sentence,
                'phrase': phrase.lower(),
                'sentence': sentence,
                'lemmatized': lemma.lower(),
            })

        if file_path is None:
            return
        with open(osp.abspath(file_path), 'w') as f:
            json.dump(data, f, indent=2)

    def prepare_data(self):
        train_file = osp.abspath(osp.join(self.root, 'train.json'))
        test_file = osp.abspath(osp.join(self.root, 'test.json'))

        if not osp.isfile(train_file):
            self.prepare_split('train', train_file)

        if not osp.isfile(test_file):
            self.prepare_split('test', test_file)

    def setup_features(self, features_path):
        features = dict()
        file_names = os.listdir(features_path)
        for file_name in file_names:
            file_path = osp.join(features_path, file_name)
            term = osp.splitext(file_name)[0].replace('_', ' ')
            features[term] = torch.load(file_path)
        return features

    def setup(self, stage='fit'):
        # load terms
        with open(osp.join(self.root, 'terms.tsv'), 'r') as f:
            lines = [line.strip().split('\t') for line in f.readlines()][1:]
            self.terms = dict()
            for (term, definition) in lines:
                self.terms[term] = definition

        percent = self.val_percent
        with open(osp.join(self.root, 'train.json'), 'r') as f:
            labeled = json.load(f)
        num_labeled_examples = len(labeled)
        num_train_examples = int((1-percent) * num_labeled_examples)
        num_val_examples = num_labeled_examples - num_train_examples
        train_data, val_data = random_split(
            labeled,
            [num_train_examples, num_val_examples],
            generator=self.torch_rng,
        )

        self.labeled_data = EuphemismDataset(labeled, 'trainval')
        self.train_data = EuphemismDataset(train_data, 'train')
        self.val_data = EuphemismDataset(val_data, 'val')
        with open(osp.join(self.root, 'test.json'), 'r') as f:
            self.test_data = EuphemismDataset(json.load(f), 'test')
        
        self.image_features = dict()
        if self.use_images:
            feature_dir = osp.join(self.root, 'download', 'features')
            self.image_features = self.setup_features(feature_dir)

        self.term_features = dict()
        self.defn_features = dict()
        if self.use_hallucinations:
            self.term_features = self.setup_features(
                osp.join(self.root, 'dalle_features', 'term'))
            self.defn_features = self.setup_features(
                osp.join(self.root, 'dalle_features', 'defn'))

    def _dataloader(self, dataset, split, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=create_collate_fn(
                split=split,
                tokenizer=self.tokenizer,
                text_input=self.text_input,
                use_definitions=self.use_definitions,
                terms=self.terms,
                use_images=self.use_images,
                image_features=self.image_features,
                use_hallucinations=self.use_hallucinations,
                term_features=self.term_features,
                defn_features=self.defn_features,
            ),
        )

    def train_dataloader(self):
        return self._dataloader(self.train_data, 'train', shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_data, 'val')

    def predict_dataloader(self):
        return [
            # self._dataloader(self.labeled_data, 'trainval'),
            self._dataloader(self.test_data, 'test'),
        ]


def create_collate_fn(
    split,
    tokenizer,
    text_input,
    use_definitions,
    terms,
    use_images,
    image_features,
    use_hallucinations,
    term_features,
    defn_features,
):
    def helper(batch, key):
        return [x.get(key) for x in batch]

    def get_sentences_with_definitions(batch):
        sentences = []
        for item in batch:
            term = item.get('lemmatized', 'unknown')
            defn = terms.get(term, 'none')
            sent = item[text_input]
            prompt = f'Term: {term}. Definition: {defn}. Sentence: {sent}'
            sentences.append(prompt)
        return sentences

    def get_features(batch, features_dict):
        features = []
        for item in batch:
            term = item.get('lemmatized')

            # FIXME: hardcoded bug fix show!
            if term == 'enhance " " interrogation technique':
                term = '"enhance "" "" interrogation technique"'
            if term == 'golden " " year':
                term = '"golden "" "" year"'

            this = features_dict.get(term)
            features.append(this)
        return torch.cat(features, dim=0)

    def _collate_fn(batch):
        indexes = torch.tensor(helper(batch, 'index'))
        if use_definitions:
            sentences = get_sentences_with_definitions(batch)
        else:
            sentences = helper(batch, text_input)
        sentences = [x.replace('@ ', '') for x in sentences]
        inputs = tokenizer(sentences, return_tensors='pt', padding=True)

        batch_image_features = None
        if use_images:
            batch_image_features = get_features(batch, image_features)
        
        batch_term_features = batch_defn_features = None
        if use_hallucinations:
            batch_term_features = get_features(batch, term_features)
            batch_defn_features = get_features(batch, defn_features)

        labels = None
        if split != 'test':
            labels = torch.tensor(helper(batch, 'label')).long()

        return {
            'indexes': indexes,
            'inputs': inputs,
            'labels': labels,
            'image_features': batch_image_features,
            'term_features': batch_term_features,
            'defn_features': batch_defn_features,
        }
    return _collate_fn


