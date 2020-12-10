"""Tasks for training the word embeddings once a corpus file has been generated from the upstream XML parse task."""

from datetime import datetime

from gensim.models import Word2Vec, FastText
from gensim.models.callbacks import CallbackAny2Vec
from luigi import Task, WrapperTask, LocalTarget

import config
from pipeline.extract_from_xml_tags import CorpusXMLToParquet


# https://stackoverflow.com/a/54423541/8857601
class MonitorCallback(CallbackAny2Vec):
    """Logging utility for monitoring the training of gensim's word embedding implementations."""
    def __init__(self):
        pass
    
    def on_train_begin(self, model):
        print(f'Training is beginning: {datetime.now()}')
        
    def on_train_end(self, model):
        print(f'Training has ended: {datetime.now()}')
        
    def on_epoch_begin(self, model):
        print(f'Epoch is beginning: {datetime.now()}')
    
    def on_epoch_end(self, model):
        print(f'Epoch has ended: {datetime.now()}')


class TrainFastText(Task):
    def requires(self):
        return [CorpusXMLToParquet()]  # because it produces the corpus file too
    
    def output(self):
        return LocalTarget(config.FASTTEXT_FILE)
    
    def run(self):
        monitor = MonitorCallback()
        print(f'Init FastText: {datetime.now()}')
        fasttext_model = FastText(**config.FASTTEXT_INIT, callbacks=[monitor])
        fasttext_model.save(str(config.FASTTEXT_FILE))


class TrainWord2Vec(Task):
    def requires(self):
        return [CorpusXMLToParquet()]  # because it produces the corpus file too
    
    def output(self):
        return LocalTarget(config.WORD2VEC_FILE)
    
    def run(self):
        monitor = MonitorCallback()
        print(f'Init Word2Vec: {datetime.now()}')
        word2vec_model = Word2Vec(**config.WORD2VEC_INIT, callbacks=[monitor])
        word2vec_model.save(str(config.WORD2VEC_FILE))


class TrainAllWordEmbeddings(WrapperTask):
    def requires(self):
        return [TrainFastText(), TrainWord2Vec()]
