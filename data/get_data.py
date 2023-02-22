import torch
from .TokenEase import Pipe
import gensim.downloader as gensim_api
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_20newsgroups


class TorchDatasetBoW(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'bow': self.data[idx]}

def get_vocab_embeddings(vocab: dict):
    # download glove.6B.50d.txt from https://nlp.stanford.edu/projects/glove/
    model = gensim_api.load('glove-wiki-gigaword-300')
    embeddings = torch.zeros(len(vocab), 300)
    for i, word in enumerate(vocab):
        if word in model:
            embeddings[i] = torch.from_numpy(model[word].copy())
    return embeddings

def get_data(data_name, batch_size):
    if data_name == '20ng':
        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        train_text, train_labels = train_data['data'], train_data['target']
        test_text, test_labels = test_data['data'], test_data['target']
        val_text, val_labels = test_text, test_labels
    else:
        raise NotImplementedError

    pipe = Pipe(preprocess=True,
                max_df=0.80,
                min_df=50,
                doc_start_token='<s>',
                doc_end_token='</s>',
                unk_token='<unk>',
                email_token='<email>',
                url_token='<url>',
                number_token='<number>',
                alpha_num_token='<alpha_num>')

    train_bow = pipe.process_data(train_text)
    test_bow, _ = pipe.get_doc_bow(test_text)
    val_bow = test_bow
    text = pipe.text
    vocab = pipe.vocab

    train_dataset = TorchDatasetBoW(train_bow)
    test_dataset = TorchDatasetBoW(test_bow)
    val_dataset = TorchDatasetBoW(val_bow)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader, text, vocab


if __name__ == '__main__':
    train_loader, test_loader, val_loader, text, vocab = get_data('20ng', 32)
    print(text[0])
