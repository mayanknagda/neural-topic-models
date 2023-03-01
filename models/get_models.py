from .dvae import DVAE
from .dvae_rsvi import DVAE_RSVI
from .etm import ETM
from .etm_dirichlet import ETMD
from .etm_dirichlet_rsvi import ETMD_RSVI
from .prod_lda import ProdLDA
from .nb_vae import NB_VAE


def get_model(model_name, vocab_size, vocab_embeddings, topic_size):
    if model_name == 'dvae':
        model = DVAE(vocab_size=vocab_size, topic_size=topic_size)
    elif model_name == 'dvae_rsvi':
        model = DVAE_RSVI(vocab_size=vocab_size, topic_size=topic_size)
    elif model_name == 'etm':
        model = ETM(vocab_size=vocab_size, vocab_embeddings=vocab_embeddings, topic_size=topic_size)
    elif model_name == 'etm_dirichlet':
        model = ETMD(vocab_size=vocab_size, vocab_embeddings=vocab_embeddings, topic_size=topic_size)
    elif model_name == 'etm_dirichlet_rsvi':
        model = ETMD_RSVI(vocab_size=vocab_size, vocab_embeddings=vocab_embeddings, topic_size=topic_size)
    elif model_name == 'prod_lda':
        model = ProdLDA(vocab_size=vocab_size, topic_size=topic_size)
    elif model_name == 'nb_vae':
        model = NB_VAE(vocab_size=vocab_size, topic_size=topic_size)
    else:
        raise NotImplementedError
    return model