import unittest
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from customer_classification.util.data_loaders import FileDataLoader

class TestDataLoader(unittest.TestCase):

    def test_check_file_exists(self):
        
        # Should not return an exception
        true_file = './data/input/final_dataset.csv'
        file_loader = FileDataLoader(true_file)
        file_loader.check_file_exists()

        # Should return an exception
        ghost_file = './data/input/ghost.csv'
        file_loader = FileDataLoader(ghost_file)
        with self.assertRaises(Exception):
            file_loader.check_file_exists()


if __name__ == "__main__":
    unittest.main()
