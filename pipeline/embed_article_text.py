"""Take the entire corpus as a Dask Dataframe and produce the document embeddings necessary for performing
   spectral clustering."""

import numpy
from luigi import Task, WrapperTask, LocalTarget, Parameter

from pipeline.extract_from_xml_tags import CorpusXMLToParquet


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

    def requires(self):
        return [CorpusXMLToParquet()]
    
    def output(self):
        return LocalTarget()
    
    def run(self):
        # TODO: load the dask dataframe from the parquet files
        # TODO: possibly adjust the index and re-partition
        # TODO: apply the embed_document function to the tokenized column and join to the ddf
        # TODO: write the augmented ddf to disk
        pass
