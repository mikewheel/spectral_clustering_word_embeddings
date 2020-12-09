"""Processes the output of the WikiExtractor module by reading XML fragments from disk, extracting text and metadata,
 and saving in batches to a series of Parquet files that can be read in by Dask for downstream tasks."""

from io import StringIO

import pandas
from luigi import Task, WrapperTask, LocalTarget, Parameter
from lxml.etree import XMLParser, parse

import config


class ExtractParsedTextFromXMLTags(Task):
    """Accepts a Path to a single text file produced by the WikiExtractor. Reads every Document tag and
       maps the ID to the raw/tokenized text before saving to """
    
    path_to_xml_fragments = Parameter()
    
    def requires(self):
        return None
    
    def output(self):
        # Mirror the subdirectory structure of the XML_DIR
        parquet_subdir = config.WIKIPEDIA_PARQUET_DIR / self.path_to_xml_fragments.parent.stem
        parquet_subdir.mkdir(parents=True, exist_ok=True)
        return LocalTarget(parquet_subdir / f'{self.path_to_xml_fragments.stem}.parquet')
    
    def run(self):
        # assert Path(self.output().path).parent.parent == config.WIKIPEDIA_PARQUET_DIR
        with open(self.path_to_xml_fragments, "r") as input_f:
            # Needs a root element to parse as XML
            xml_str = '<root>\n' + input_f.read() + '\n</root>'
            
            # Handle malformed XML from WikiExtractor: https://stackoverflow.com/a/9050454/8857601
            parser = XMLParser(encoding='utf-8', recover=True, remove_blank_text=True)
            tree = parse(StringIO(xml_str), parser=parser)
            root = tree.getroot()
        
        doc_rows = []
        
        # For each child tag under the newly-constructed root tag
        for doc_tag in root.findall('doc'):
            # Extract target information from this tag
            doc_id = doc_tag.attrib.get("id")
            doc_url = doc_tag.attrib.get("url")
            doc_title = doc_tag.attrib.get("title")
            doc_text = doc_tag.text.strip()
            # TODO strip the text of whitespace, tokenize, word embed?
            doc_rows.append([doc_id, doc_url, doc_title, doc_text])
            
        # Construct a dataframe, and then construct parquet file output
        df = pandas.DataFrame(doc_rows, columns=["id", "url", "title", "text"])
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
        return [ExtractParsedTextFromXMLTags(path_to_xml_fragments=filepath) for filepath in text_file_list]
