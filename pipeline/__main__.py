from luigi import build

from pipeline.extract_from_xml_tags import CorpusXMLToParquet
from pipeline.perform_clustering import PerformSpectralClustering


if __name__ == "__main__":
    tasks = [CorpusXMLToParquet()]
    build(tasks, local_scheduler=True, workers=8)  # TODO set up central scheduler
