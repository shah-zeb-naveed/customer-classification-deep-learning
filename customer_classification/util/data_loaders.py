import logging
from abc import ABC, abstractmethod
import os.path
import os
import pandas as pd

class AbstractDataLoader(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_data(self, filename):
        logging.info('Checking file exists.')

        if not os.path.isfile(filename):
            logging.error('File does not exist')
            raise Exception("File does not exist")
        else:
            logging.info('Found file: ' + filename)

class FileDataLoader(AbstractDataLoader):

    # Initialization
    def __init__(self, filename: str):
        super().__init__()
        logging.info('Initializing Data Loading')
        self.filename = filename
    
    def check_file_exists(self):
        if not os.path.isfile(self.filename):
            logging.error('File does not exist')
            raise Exception("File \"{}\" does not exist".format(self.filename))
        else:
            logging.info('Found file: ' + self.filename)

    # Load data from file and return data
    def load_data(self):
        # Check file exists
        logging.info('Checking file exists.')
        self.check_file_exists()

        # Load data
        logging.info('Loading data using pandas')
        df = pd.read_csv(self.filename)

        return df

