import models
import data
from utils import TopicEval
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser


def main(settings):
    # set seed
    seed_everything(42, workers=True)
    # import data
    train_loader, test_loader, val_loader, text, vocab = data.get_data(settings['data_name'], settings['batch_size'])
    # import model
    model = models.get_model(model_name=settings['model_name'],
                             vocab_size=len(vocab),
                             topic_size=settings['topic_size'])
    # logger
    logger = TensorBoardLogger(settings['root_dir'], name=settings['model_name'], version=settings['data_name'])
    # train
    trainer = Trainer(max_epochs=settings['max_epochs'],
                      accelerator=settings['accelerator'],
                      devices=settings['devices'],
                      default_root_dir=settings['root_dir'],
                      logger=logger)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # get topics and evaluate
    best_model_path = trainer.checkpoint_callback.best_model_path
    topics = model.get_topics(vocab=vocab, path=best_model_path)
    eval = TopicEval(vocab=vocab, text=text)
    c_v = eval.topic_coherence(metric='c_v', topics=topics)
    c_npmi = eval.topic_coherence(metric='c_npmi', topics=topics)
    td = eval.topic_diversity(topics=topics)
    # save all results in .txt file
    tensor_board = trainer.logger.experiment
    tensor_board.add_text('c_v', str(c_v))
    tensor_board.add_text('c_npmi', str(c_npmi))
    tensor_board.add_text('td', str(td))
    path = trainer.log_dir + '/topics.txt'
    with open(path, 'w') as f:
        f.write('c_v: ' + str(c_v) + '\t' + 'c_npmi: ' + str(c_npmi) + '\t' + 'td: ' + str(td) + '\n')
        for i, topic in enumerate(topics):
            f.write('Topic ' + str(i+1) + ': ' + ' '.join(topic) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, default='20ng')
    parser.add_argument('--model_name', type=str, default='dvae')
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--root_dir', type=str, default='output/')
    parser.add_argument('--devices', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--topic_size', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=100)
    args = parser.parse_args()
    settings = vars(args)
    if settings['devices'] == -1:
        settings['devices'] = 'auto'
    main(settings)