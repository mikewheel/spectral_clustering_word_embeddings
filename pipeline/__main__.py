from luigi import build

from pipeline.extract_from_xml_tags import CorpusXMLToParquet
from pipeline.perform_clustering import PerformSpectralClustering


if __name__ == "__main__":
    tasks = [CorpusXMLToParquet(), PerformSpectralClustering(num_clusters=10)]
    build(tasks)  # TODO set up scheduler, dashboard
