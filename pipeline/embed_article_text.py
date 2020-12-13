"""Tasks for taking the entire corpus as a Dask Dataframe and producing the document embeddings necessary for performing
   spectral clustering downstream."""

import dask.dataframe
import numpy
import pandas
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
    # Fetch pre-loaded word vectors
    model = models[model_name]
    
    # Collect all word vectors in the article text
    word_vectors = []
    doc_tokens = [token for sent in doc_sentences for token in sent]
    for word in doc_tokens:
        try:
            word_vectors.append(model[word])
        except KeyError:
            word_vectors.append(numpy.zeros(config.WORD_VECTOR_SIZE))
    
    if len(word_vectors) == 0:
        return numpy.zeros(config.WORD_VECTOR_SIZE * 2)
    
    # Aggregate word vectors for document with min and max
    word_vectors = numpy.array(word_vectors)
    try:
        el_min = numpy.amin(word_vectors, axis=0)
        el_max = numpy.amax(word_vectors, axis=0)
        out = numpy.concatenate((el_min, el_max))
        assert out.shape[0] == config.WORD_VECTOR_SIZE * 2, out.shape
    except ValueError as e:
        out = numpy.zeros(config.WORD_VECTOR_SIZE * 2)
    
    return out


def embed_partition(df, model_name):
    doc_vectors = df["tokenized_text"].apply(embed_single_document, model_name=model_name)
    doc_vectors_df = pandas.DataFrame.from_records(doc_vectors.values)
    out_df = pandas.concat([df, doc_vectors_df], axis=1)
    # https://stackoverflow.com/a/38577252/8857601
    out_df.columns = out_df.columns.astype(str)
    return out_df


class GenerateDocumentEmbeddings(Task):
    """Map over the partitions of our Dask dataframe."""
    
    model = Parameter()

    def requires(self):
        return [TrainAllWordEmbeddings()]
    
    def output(self):
        return LocalTarget(config.ARTICLE_EMBEDDINGS_DIR / f'{self.model}_csv_success.txt')
    
    def run(self):
        # Load the dask dataframe from the parquet files
        ddf = dask.dataframe.read_parquet(config.WIKIPEDIA_PARQUET_DIR / "*" / "wiki_*.parquet")

        # Apply the document embedding strategy to each partition in the dask dataframe
        meta_dtypes = [('id', ddf['id'].dtype), ('url', ddf['url'].dtype), ('title', ddf['title'].dtype),
                       ('tokenized_text', ddf['tokenized_text'].dtype)] + \
            [(str(i), 'f8') for i in range(config.WORD_VECTOR_SIZE * 2)]
        ddf = ddf.map_partitions(embed_partition, model_name=self.model, meta=meta_dtypes)
        ddf = ddf.drop(["tokenized_text"], axis=1)
        
        # Write the augmented ddf to disk
        ddf_output_path = config.ARTICLE_EMBEDDINGS_DIR / f'{self.model}_to_csv'
        ddf_output_path.mkdir(parents=True, exist_ok=True)
        dask.dataframe.to_csv(ddf, ddf_output_path)
        
        with self.output().open("w") as f:
            f.write(f'Document embedding with {self.model}: SUCCESS')
    

class GenerateAllDocumentEmbeddings(WrapperTask):
    def requires(self):
        return [GenerateDocumentEmbeddings(model=m) for m in {"word2vec", "fasttext"}]
