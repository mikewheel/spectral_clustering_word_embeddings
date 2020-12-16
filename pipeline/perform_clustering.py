"""Assembles a Dask Dataframe from the Parquet files created by upstream tasks and performs Spectral Clustering
   experiments on the data."""

import dask.dataframe
from dask import compute
from dask_ml.cluster import SpectralClustering, KMeans
from datetime import datetime
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

        print(f'Initializing dask dataframe of word embeddings at {datetime.now()}')
        ddf = dask.dataframe.read_csv(config.ARTICLE_EMBEDDINGS_DIR / f'{self.word_vectors}_to_csv' / "*.part")

        print(f'Dropping columns and converting to design matrix (dask array) at {datetime.now()}')
        X = ddf.drop(['Unnamed: 0', "id", "url", "title"], axis=1)
        X = X.to_dask_array(lengths=True)
            
        # Perform k-means clustering
        print(f'Starting K-Means clustering at {datetime.now()}')
        k_means_clustering_model = KMeans(n_clusters=self.num_clusters, n_jobs=-1, max_iter=config.K_MEANS_MAX_ITER)
        k_means_cluster_labels = k_means_clustering_model.fit(X)
        
        # Write k-means results to disk
        print(f'Joining K-means results and writing to disk at {datetime.now()}')
        k_means_results_ddf = ddf.join(k_means_cluster_labels)
        k_means_ddf_output_path = config.CLUSTERING_RESULTS_DIR / f'{self.word_vectors}_w_k_means'
        k_means_ddf_output_path.mkdir(parents=True, exist_ok=True)
        dask.dataframe.to_csv(k_means_results_ddf, k_means_ddf_output_path)
        
        # Perform spectral clustering
        print(f'Starting Spectral clustering at {datetime.now()}')
        spectral_clustering_model = SpectralClustering(n_clusters=self.num_clusters, n_jobs=-1,
                                                       persist_embedding=True,
                                                       kmeans_params={"max_iter": config.K_MEANS_MAX_ITER})
        spectral_cluster_labels = spectral_clustering_model.fit(X)
        
        # Write spectral results to disk
        print(f'Joining Spectral results and writing to disk at {datetime.now()}')
        spectral_results_ddf = ddf.join(spectral_cluster_labels)
        spectral_ddf_output_path = config.CLUSTERING_RESULTS_DIR / f'{self.word_vectors}_w_spectral'
        spectral_ddf_output_path.mkdir(parents=True, exist_ok=True)
        dask.dataframe.to_csv(spectral_results_ddf, spectral_ddf_output_path)
        
        # And save the success flag
        with self.output().open("w") as f:
            # f.write(f'Clustering {self.word_vectors} k={self.num_clusters}: {silhouette_score_result}' + "\n")
            # f.write(spectral_clustering_model.get_params(deep=True))
            f.write(f'{self.word_vectors}: Success!')
