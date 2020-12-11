"""Assembles a Dask Dataframe from the Parquet files created by upstream tasks and performs Spectral Clustering
   experiments on the data."""

import dask.dataframe
from dask_ml.cluster import SpectralClustering
from luigi import Task, LocalTarget, IntParameter, Parameter

import config
from pipeline.embed_article_text import GenerateDocumentEmbeddings


class PerformSpectralClustering(Task):
    """Coalesce the Wikipedia article parquet files as a Dask Dataframe, and perform spectral clustering using
       the magic of Dask-ML."""
    
    num_clusters = IntParameter()
    word_vectors = Parameter()
    
    def requires(self):
        return [GenerateDocumentEmbeddings(model=self.word_vectors)]

    def output(self):
        return LocalTarget(config.CLUSTERING_RESULTS_DIR / f'cluster_{self.num_clusters}_{self.word_vectors}.txt')
    
    def run(self):
        if self.word_vectors not in {"fasttext", "word2vec"}:
            raise ValueError(f'Expected fasttext or word2vec; got {self.word_vectors}')
        
        ddf = dask.dataframe.read_parquet(config.ARTICLE_EMBEDDINGS_DIR / self.word_vectors)
        
        # Perform spectral clustering on the vectorized word columns only
        clustering_model = SpectralClustering(n_clusters=self.num_clusters, n_jobs=-1, persist_embedding=True)
        cluster_labels = clustering_model.fit(ddf.drop(["id", "url", "title", "tokenized_text"], axis=1))
        
        # Attach cluster labels to ddf and write the augmented ddf to disk
        ddf = ddf.join(cluster_labels)
        ddf_output_path = config.CLUSTERING_RESULTS_DIR / self.word_vectors
        ddf_output_path.mkdir(parents=True, exist_ok=True)
        dask.dataframe.to_parquet(ddf, ddf_output_path)
        
        # And save the parameters of the spectral clustering object to the target
        with self.output().open("w") as f:
            f.write(clustering_model.get_params(deep=True))
