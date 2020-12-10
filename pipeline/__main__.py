from luigi import build

from pipeline.extract_from_xml_tags import CorpusXMLToParquet
from pipeline.embed_article_text import GenerateDocumentEmbeddings
from pipeline.perform_clustering import PerformSpectralClustering

if __name__ == "__main__":
    tasks = [CorpusXMLToParquet()]
    build(tasks, local_scheduler=True)
