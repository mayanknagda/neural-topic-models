from .dvae import DVAE
from .dvae_rsvi import DVAE_RSVI


def get_model(model_name, vocab_size, topic_size):
    if model_name == 'dvae':
        model = DVAE(vocab_size=vocab_size, topic_size=topic_size)
    elif model_name == 'dvae_rsvi':
        model = DVAE_RSVI(vocab_size=vocab_size, topic_size=topic_size)
    else:
        raise NotImplementedError
    return model