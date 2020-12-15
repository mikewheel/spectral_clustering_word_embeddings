"""Assembles a Dask Dataframe from the Parquet files created by upstream tasks and performs Spectral Clustering
   experiments on the data."""

import dask.dataframe
from dask import compute
from dask_ml.cluster import SpectralClustering, KMeans
from luigi import Task, LocalTarget, IntParameter, Parameter
from sklearn.metrics import silhouette_score

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
        
        ddf = dask.dataframe.read_csv(config.ARTICLE_EMBEDDINGS_DIR / f'{self.word_vectors}_to_csv' / "*.part")
        
        # Look at the memory usage of each partition to decide if we can put this into pandas directly
        memory_usage = ddf.memory_usage_per_partition(deep=True)
        
        for part_idx, val in memory_usage.iteritems():
            print(f'Partition index: {part_idx}, memory usage: {val}')
        
        X = ddf.drop(["id", "url", "title"], axis=1)
        print(X.columns)
        X = X.to_dask_array(lengths=True)
            
        # Perform k-means clustering
        k_means_clustering_model = KMeans(n_clusters=self.num_clusters, n_jobs=-1)
        k_means_cluster_labels = k_means_clustering_model.fit(X)
        # Write k-means results to disk
        k_means_results_ddf = ddf.join(k_means_cluster_labels)
        k_means_ddf_output_path = config.CLUSTERING_RESULTS_DIR / f'{self.word_vectors}_w_k_means'
        k_means_ddf_output_path.mkdir(parents=True, exist_ok=True)
        dask.dataframe.to_parquet(k_means_results_ddf, k_means_ddf_output_path)
        
        # Perform spectral clustering
        spectral_clustering_model = SpectralClustering(n_clusters=self.num_clusters, n_jobs=-1, persist_embedding=True)
        spectral_cluster_labels = spectral_clustering_model.fit(X)
        # Write spectral results to disk
        spectral_results_ddf = ddf.join(spectral_cluster_labels)
        spectral_ddf_output_path = config.CLUSTERING_RESULTS_DIR / f'{self.word_vectors}_w_spectral'
        spectral_ddf_output_path.mkdir(parents=True, exist_ok=True)
        dask.dataframe.to_parquet(spectral_results_ddf, spectral_ddf_output_path)
        
        # # TODO: move this to a different task so the clustering can succeed independent of the evaluation
        # # FIXME: will sklearn's function work on dask objects?
        # silhouette_score_result = silhouette_score(X, spectral_cluster_labels)
        # print(f'Clustering {self.word_vectors} k={self.num_clusters}: {silhouette_score_result}')
        
        # And save the parameters of the spectral clustering object to the target
        with self.output().open("w") as f:
            # f.write(f'Clustering {self.word_vectors} k={self.num_clusters}: {silhouette_score_result}' + "\n")
            # f.write(spectral_clustering_model.get_params(deep=True))
            f.write(f'{self.word_vectors}: Success!')
