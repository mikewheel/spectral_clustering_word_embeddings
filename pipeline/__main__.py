from luigi import build

from pipeline.embed_article_text import GenerateDocumentEmbeddings
from pipeline.perform_clustering import PerformSpectralClustering

if __name__ == "__main__":
    
    tasks = [GenerateDocumentEmbeddings()]
    build(tasks, local_scheduler=True)
