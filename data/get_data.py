from .TokenEase import Pipe
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_20newsgroups


class TorchDatasetBoW(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'bow': self.data[idx]}


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
                max_df=0.90,
                min_df=20,
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
