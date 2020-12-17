# DS5230: Final Project
A comparison of word embedding algorithms applied to the clustering of English Wikipedia. 
Written by Michael Wheeler, December 2020.

## Project Structure
  - `config.py`: Contains project-wide configurations, mainly paths to data assets and model hyperparameters as 
     keyword arguments to be passed on initialization
  - `data/`: Contains the data assets for the project, such as the multistream bzip2 archives containing Wikipedia. 
  - `pipeline/`: Contains the actual procedure to execute the project steps from (almost) the beginning to the end. The 
     pipeline dependencies can be traced back by looking at the `requires` method for each Task. For more information
     see [the Luigi docs](https://luigi.readthedocs.io/en/stable/).
  - `report/`: Contains the LaTeX and LyX editor files I used to generate the final report. Also contains the final 
     report itself as a PDF.

## Setup
  - Install Python 3.7.9 or higher, `sudo yum install gcc python3-devel`
  - `python3 -m venv virutalenv`, `source virtualenv/bin/activate`, `pip install -r requirements.txt`
  - Download Wikipedia archives to the project's data directory: for detailed instructions see 
    [this Wikipedia page](https://en.wikipedia.org/wiki/Wikipedia:Database_download)
  - In order to run the WikiExtractor itself (NOT included in the pipeline) follow the instructions 
    [here](https://github.com/attardi/wikiextractor/issues/222)
  - Finally, activate the pipeline: `python3 -m pipeline`
