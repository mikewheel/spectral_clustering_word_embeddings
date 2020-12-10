from luigi import build

from pipeline.train_word_embeddings import TrainWordEmbeddings
from pipeline.embed_article_text import GenerateDocumentEmbeddings
from pipeline.perform_clustering import PerformSpectralClustering

if __name__ == "__main__":
    
    tasks = [TrainWordEmbeddings()]
    build(tasks, local_scheduler=True)
