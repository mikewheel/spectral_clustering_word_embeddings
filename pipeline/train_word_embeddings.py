"""Tasks for training the word embeddings once a corpus file has been generated from the upstream XML parse task."""

from gensim.models import Word2Vec, FastText
from luigi import Task, WrapperTask, LocalTarget

import config
from pipeline.extract_from_xml_tags import CorpusXMLToParquet


class TrainFastText(Task):
    def requires(self):
        return [CorpusXMLToParquet()]  # because it produces the corpus file
    
    def output(self):
        return LocalTarget(config.FASTTEXT_FILE)
    
    def run(self):
        fasttext_model = FastText(**config.FASTTEXT_INIT)
        fasttext_model.save(str(config.FASTTEXT_FILE))


class TrainWord2Vec(Task):
    def requires(self):
        return [CorpusXMLToParquet()]  # because it produces the corpus file too
    
    def output(self):
        return LocalTarget(config.WORD2VEC_FILE)
    
    def run(self):
        word2vec_model = Word2Vec(**config.WORD2VEC_INIT)
        word2vec_model.save(str(config.WORD2VEC_FILE))


class TrainWordEmbeddings(WrapperTask):
    def requires(self):
        return [TrainFastText(), TrainWord2Vec()]
