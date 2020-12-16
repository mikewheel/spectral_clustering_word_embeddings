from luigi import build

from pipeline.embed_article_text import GenerateAllDocumentEmbeddings
from pipeline.perform_clustering import PerformSpectralClustering

if __name__ == "__main__":
    
    tasks = [GenerateAllDocumentEmbeddings(), PerformSpectralClustering(num_clusters=2, word_vectors="fasttext"),
             PerformSpectralClustering(num_clusters=2, word_vectors="word2vec")]
    build(tasks, local_scheduler=True)
