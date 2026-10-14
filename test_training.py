import csv
from pathlib import Path
import tempfile
import unittest
from train import load_rows

class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'fixture.csv'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows, fields=('Resume_str','Category')):
        with self.path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_deduplication_and_missing_rows(self):
        self.write([['<p>Python SQL</p>','IT'], [' python   sql ','IT'], ['', 'IT'], ['Accounting',' ']])
        rows, summary = load_rows(self.path, return_summary=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary['input_rows'], 4)
        self.assertEqual(summary['skipped_rows'], 2)
        self.assertEqual(summary['exact_normalized_duplicates_removed'], 1)
        self.assertEqual(len(summary['source_sha256']), 64)

    def test_conflicting_labels_are_rejected(self):
        self.write([['Python SQL','IT'], ['python sql','FINANCE']])
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            load_rows(self.path)

    def test_unlabeled_files_are_rejected(self):
        self.write([['fixture.txt','Python SQL']], fields=('file_name','text'))
        with self.assertRaisesRegex(ValueError, 'Category'):
            load_rows(self.path)

if __name__ == '__main__':
    unittest.main()
