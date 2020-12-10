from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = Path("/data")

# Original archive files downloaded from Wikipedia
WIKIPEDIA_ARCHIVE_CONTENTS_FILE = DATA_DIR / "enwiki-20201101-pages-articles-multistream.xml.bz2"
WIKIPEDIA_ARCHIVE_INDEX_FILE = DATA_DIR / "enwiki-20201101-pages-articles-multistream-index.txt.bz2"

# Directory for WikiExtractor output: subdirectories contain text files containing XML fragments for each article
WIKIPEDIA_XML_DIR = DATA_DIR / "archive_xml"
# Directory for parquet files containing article text: structure mirrors XML_DIR, maps each text file to a parquet file
WIKIPEDIA_PARQUET_DIR = DATA_DIR / "archive_parquets"
# Directory for storing clustered data, summary statistics, and plots
CLUSTERING_RESULTS_DIR = DATA_DIR / "clustering_results"

LINE_SENTENCE_CORPUS_FILE = DATA_DIR / "line_sentence_corpus.txt"

WORD_VECTOR_SIZE = 200
FASTTEXT_INIT = {"size": WORD_VECTOR_SIZE, "window": 4, "min_count": 1, "corpus_file": str(LINE_SENTENCE_CORPUS_FILE),
                 "workers": 8}
WORD2VEC_INIT = {"size": WORD_VECTOR_SIZE, "window": 4, "min_count": 1, "corpus_file": str(LINE_SENTENCE_CORPUS_FILE),
                 "workers": 8}
WORD_VECTOR_DIR = DATA_DIR / "word_vectors"
FASTTEXT_FILE = WORD_VECTOR_DIR / "fasttext.model"
WORD2VEC_FILE = WORD_VECTOR_DIR / "word2vec.model"
