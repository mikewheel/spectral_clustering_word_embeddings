"""Tasks for taking the entire corpus as a Dask Dataframe and producing the document embeddings necessary for performing
   spectral clustering downstream."""

import dask.dataframe
import numpy
import pandas
from contextlib import suppress
from datetime import datetime
from luigi import Task, WrapperTask, LocalTarget, Parameter
from gensim.models import Word2Vec, FastText

import config
from pipeline.train_word_embeddings import TrainAllWordEmbeddings

models = {
    "fasttext": FastText.load(str(config.FASTTEXT_FILE)).wv,
    "word2vec": Word2Vec.load(str(config.WORD2VEC_FILE)).wv,
}


def embed_single_document(doc_sentences, model_name):
    """Given a list of list of tokens and a word vector model, return the vector representation of the document."""
    print(f'{datetime.now()}: Entering call to embed_single_document')
    # Fetch pre-loaded word vectors
    model = models[model_name]
    
    # Collect all word vectors in the article text
    word_vectors = []
    doc_tokens = [token for sent in doc_sentences for token in sent]
    for word in doc_tokens:
        with suppress(KeyError):  # Just ignore any any out-of-vocab words
            word_vectors.append(model[word])

    # Aggregate word vectors for document with min and max
    word_vectors = numpy.array(word_vectors)
    assert word_vectors.shape[0] == len(doc_tokens)
    el_min = numpy.amin(word_vectors, axis=0)
    el_max = numpy.amax(word_vectors, axis=0)
    out = numpy.concatenate([el_min, el_max], axis=1)
    print(f'{datetime.now()}: Exiting call to embed_single_document. out {out.shape}')
    return out


def embed_partition(df, model_name):
    print(f'{datetime.now()}: Enter call to embed_partition. df {df.shape}')
    doc_vectors = df["tokenized_text"].apply(embed_single_document, axis=1, model_name=model_name)
    print(f'{datetime.now()}: Call to df.apply complete. doc_vectors {doc_vectors.shape}')
    out_df = pandas.concat([df, doc_vectors], axis=1)
    print(f'{datetime.now()}: Exiting call to embed_partition. out_df {out_df.shape}')
    return out_df


class GenerateDocumentEmbeddings(Task):
    """Map over the partitions of our Dask dataframe."""
    
    model = Parameter()

    def requires(self):
        return [TrainAllWordEmbeddings()]
    
    def output(self):
        return LocalTarget(config.ARTICLE_EMBEDDINGS_DIR / f'{self.model}_success.txt')
    
    def run(self):
        print(f'{datetime.now()}: Reading Dask Dataframe from Parquet')
        # Load the dask dataframe from the parquet files
        ddf = dask.dataframe.read_parquet(config.WIKIPEDIA_PARQUET_DIR / "*" / "wiki_*.parquet")

        print(f'{datetime.now()}: Setting ID column as index')
        # Set the article ID to be the dataframe's index
        ddf = ddf.set_index("id", npartitions="auto")

        print(f'{datetime.now()}: Call to map_partitions')
        # Apply the document embedding strategy to each partition in the dask dataframe
        ddf = ddf.map_partitions(embed_partition, model_name=self.model)

        print(f'{datetime.now()}: Writing out to parquet')
        # Write the augmented ddf to disk
        ddf_output_path = config.ARTICLE_EMBEDDINGS_DIR / self.model
        ddf_output_path.mkdir(parents=True, exist_ok=True)
        dask.dataframe.to_parquet(ddf, ddf_output_path, overwrite=True)
        
        with self.output().open("w") as f:
            f.write(f'Document embedding with {self.model}: SUCCESS')
    

class GenerateAllDocumentEmbeddings(WrapperTask):
    def requires(self):
        return [GenerateDocumentEmbeddings(model=m) for m in {"word2vec", "fasttext"}]
