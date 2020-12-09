"""Assembles a Dask Dataframe from the Parquet files creaated by upstream tasks and performs Spectral Clustering
   experiments on the data."""

import config
import dask.dataframe
from dask_ml.cluster import SpectralClustering
import xml.etree.ElementTree as ElementTree

from itertools import chain
from luigi import Task, WrapperTask, LocalTarget, Parameter, build, IntParameter
from pathlib import Path

from pipeline.extract_from_xml_tags import CorpusXMLToParquet


class PerformSpectralClustering(Task):
    """Coalesce the Wikipedia article parquet files as a Dask Dataframe, and perform spectral clustering using
       the magic of Dask-ML."""
    
    num_clusters = IntParameter()
    
    def requires(self):
        return [CorpusXMLToParquet()]

    def output(self):
        return LocalTarget(config.CLUSTERING_RESULTS_DIR / "output.txt")  # FIXME actual target
    
    def run(self):
        ddf = dask.dataframe.read_parquet(config.WIKIPEDIA_PARQUET_DIR / "*" / "wiki_*.parquet")
        
        # TODO set ID column as index, drop features except the document vectors
        
        # Perform spectral clustering
        clustering_model = SpectralClustering(n_clusters=self.num_clusters)
        clustered_ddf = clustering_model.fit(ddf)
        
        # TODO reattach other columns to the ddf, write to disk, return
        
        with self.output().open("w") as f:
            f.write("Success!")
