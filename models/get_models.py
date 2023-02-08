from .dvae import DVAE


def get_model(model_name, vocab_size, topic_size):
    if model_name == 'dvae':
        model = DVAE(vocab_size=vocab_size, topic_size=topic_size)
    else:
        raise NotImplementedError
    return model