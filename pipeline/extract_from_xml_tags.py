"""Processes the output of the WikiExtractor module by reading XML fragments from disk, extracting text and metadata,
 and saving in batches to a series of Parquet files that can be read in by Dask for downstream tasks."""

from io import StringIO

import pandas
from gensim.models import Word2Vec, FastText
from luigi import Task, WrapperTask, LocalTarget, Parameter
from lxml.etree import XMLParser, parse
from nltk import sent_tokenize, word_tokenize

import config


class ExtractParsedTextFromXMLTags(Task):
    """Accepts a Path to a single text file produced by the WikiExtractor. Reads every Document tag and
       maps the ID to the tokenized text before saving to disk. Also generates a corpus file for the training
       of word embedding models downstream."""
    
    path_to_xml_fragments = Parameter()
    path_to_fasttext = Parameter()
    path_to_word2vec = Parameter()
    
    def requires(self):
        return None
    
    def output(self):
        # Mirror the subdirectory structure of the XML_DIR
        parquet_subdir = config.WIKIPEDIA_PARQUET_DIR / self.path_to_xml_fragments.parent.stem
        parquet_subdir.mkdir(parents=True, exist_ok=True)
        return LocalTarget(parquet_subdir / f'{self.path_to_xml_fragments.stem}.parquet')
    
    def get_fasttext(self):
        if not self.path_to_fasttext.exists():
            model = FastText(**config.FASTTEXT_INIT)
            return model, False
        else:
            model = FastText.load(str(self.path_to_fasttext))
            return model, True
            
    def get_word2vec(self):
        if not self.path_to_word2vec.exists():
            model = Word2Vec(**config.WORD2VEC_INIT)
            return model, False
        else:
            model = Word2Vec.load(str(self.path_to_word2vec))
            return model, True
    
    def run(self):
        with open(self.path_to_xml_fragments, "r") as input_f:
            # Needs a root element to parse as XML
            xml_str = '<root>\n' + input_f.read() + '\n</root>'
            
            # Handle malformed XML from WikiExtractor: https://stackoverflow.com/a/9050454/8857601
            parser = XMLParser(encoding='utf-8', recover=True, remove_blank_text=True)
            tree = parse(StringIO(xml_str), parser=parser)
            root = tree.getroot()
        
        # For dataframe construction below
        doc_rows = []
        # Gives us the word embedding models and whether or not they're already partially trained
        fasttext_model, fasttext_update = self.get_fasttext()
        word2vec_model, word2vec_update = self.get_word2vec()

        # For each child tag under the newly-constructed root tag
        for doc_tag in root.findall('doc'):
            # Extract target information from this tag
            doc_id = doc_tag.attrib.get("id")
            doc_url = doc_tag.attrib.get("url")
            doc_title = doc_tag.attrib.get("title")
            doc_text = doc_tag.text.strip()
            
            tokenized_text = [word_tokenize(sent) for sent in sent_tokenize(doc_text)]
            
            # Train fasttext on current text
            fasttext_model.build_vocab(sentences=tokenized_text, update=fasttext_update)
            fasttext_model.train(sentences=tokenized_text, total_examples=len(tokenized_text),
                                 epochs=fasttext_model.epochs)
            fasttext_update = True
            
            # Train word2vec on current text
            word2vec_model.build_vocab(sentences=tokenized_text, update=word2vec_update)
            word2vec_model.train(sentences=tokenized_text, total_examples=len(tokenized_text),
                                 epochs=word2vec_model.epochs)
            word2vec_update = True
            
            doc_rows.append([doc_id, doc_url, doc_title, tokenized_text])

        # Save the updated word vector models to disk
        fasttext_model.save(str(self.path_to_fasttext))
        word2vec_model.save(str(self.path_to_word2vec))
            
        # Construct a dataframe, and then construct parquet file output
        df = pandas.DataFrame(doc_rows, columns=["id", "url", "title", "tokenized_text"])
        df.to_parquet(open(self.output().path, "wb"))
    

class CorpusXMLToParquet(WrapperTask):
    """Applies the XML tag text extraction to each text file in each subdirectory of the WikiExtractor output."""
    
    def requires(self):
        text_file_list = [filepath  # Get the Path objects
                          # For each subdirectory under the WikiExtractor archive directory
                          for subdir in [subd_ for subd_ in config.WIKIPEDIA_XML_DIR.iterdir() if subd_.is_dir()]
                          # For each text file under the subdirectory
                          for filepath in subdir.iterdir() if filepath.is_file()]
        
        # Then convert to the text extraction Task and return
        return [ExtractParsedTextFromXMLTags(path_to_xml_fragments=filepath, path_to_fasttext=config.FASTTEXT_FILE,
                                             path_to_word2vec=config.WORD2VEC_FILE)
                for filepath in text_file_list]
