"""Take the entire corpus as a Dask Dataframe and produce the document embeddings necessary for performing
   spectral clustering."""

import dask.dataframe
import numpy
from luigi import Task, LocalTarget

import config
from pipeline.train_word_embeddings import TrainAllWordEmbeddings


def embed_document(doc_sents, model):
    """Given a list of list of tokens and a word vector model, return the vector representation of the document."""
    word_vectors = []
    doc_tokens = [token for sent in doc_sents for token in sent]
    for word in doc_tokens:
        word_vectors.append(model[word])

    # Aggregate word vectors for document with min and max
    word_vectors = numpy.array(word_vectors)
    assert word_vectors.shape[0] == len(doc_tokens)
    el_min = numpy.amin(word_vectors, axis=0)
    el_max = numpy.amax(word_vectors, axis=0)
    out = numpy.concatenate([el_min, el_max], axis=1)
    return out


class GenerateDocumentEmbeddings(Task):
    """Map over the partitions of our Dask dataframe """

    def requires(self):
        return [TrainAllWordEmbeddings()]
    
    def output(self):
        return LocalTarget()  # TODO
    
    def run(self):
        # Load the dask dataframe from the parquet files
        ddf = dask.dataframe.read_parquet(config.WIKIPEDIA_PARQUET_DIR / "*" / "wiki_*.parquet")
        # TODO: possibly adjust the index and re-partition
        # TODO: apply the embed_document function to the tokenized column and join to the ddf
        # TODO: write the augmented ddf to disk
        pass
