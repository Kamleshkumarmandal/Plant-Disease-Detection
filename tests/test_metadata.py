from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from plant_disease_labels import IDX_TO_CLASSES


ROOT = Path(__file__).resolve().parents[1]


class PlantDiseaseMetadataTests(unittest.TestCase):
    def test_class_mapping_has_39_contiguous_indices(self):
        self.assertEqual(len(IDX_TO_CLASSES), 39)
        self.assertEqual(sorted(IDX_TO_CLASSES), list(range(39)))
        self.assertEqual(IDX_TO_CLASSES[4], "Background_without_leaves")

    def test_disease_information_covers_all_model_classes(self):
        info = pd.read_csv(ROOT / "disease_info.csv", encoding="cp1252")
        self.assertEqual(len(info), 39)
        self.assertEqual(info["index"].astype(int).tolist(), list(range(39)))
        self.assertTrue(info["disease_name"].notna().all())

    def test_every_class_has_a_label(self):
        for index in range(39):
            self.assertTrue(IDX_TO_CLASSES[index].strip())


if __name__ == "__main__":
    unittest.main()
