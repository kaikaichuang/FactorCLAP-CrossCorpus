import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from prepare_crop_features import prepare_source


class CropFeaturePreparationTests(unittest.TestCase):
    def test_only_long_rows_are_replaced_by_exact_center_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_root = root / "splits"
            feature_root = root / "features"
            output_root = root / "output"
            (split_root / "msp").mkdir(parents=True)
            feature_root.mkdir()
            short_audio = root / "short.wav"
            long_audio = root / "long.wav"
            short_audio.touch()
            long_audio.touch()
            pd.DataFrame(
                {
                    "sample_id": ["short", "long"],
                    "audio_path": [str(short_audio), str(long_audio)],
                }
            ).to_csv(split_root / "msp" / "train.csv", index=False)
            pd.DataFrame(
                {
                    "sample_id": ["short", "long"],
                    "feature": [1.0, 2.0],
                    "Duration": [3.0, 9.0],
                }
            ).to_csv(
                feature_root / "msp_train_eGeMAPSv02.csv",
                index=False,
            )
            index = pd.MultiIndex.from_arrays(
                [
                    [str(long_audio)],
                    pd.to_timedelta([2.0], unit="s"),
                    pd.to_timedelta([7.0], unit="s"),
                ],
                names=["file", "start", "end"],
            )
            extracted = pd.DataFrame({"feature": [9.0]}, index=index)
            smile = mock.Mock()
            smile.process_files.return_value = extracted
            with mock.patch(
                "prepare_crop_features.opensmile.Smile",
                return_value=smile,
            ):
                prepare_source(
                    "msp",
                    split_root,
                    feature_root,
                    output_root,
                    num_workers=1,
                )

            output = pd.read_csv(
                output_root / "msp_train_eGeMAPSv02.csv"
            ).set_index("sample_id")
            self.assertEqual(output.loc["short", "feature"], 1.0)
            self.assertEqual(output.loc["short", "Duration"], 3.0)
            self.assertEqual(output.loc["long", "feature"], 9.0)
            self.assertEqual(output.loc["long", "Duration"], 5.0)
            _, kwargs = smile.process_files.call_args
            self.assertEqual(kwargs["starts"], [2.0])
            self.assertEqual(kwargs["ends"], [7.0])


if __name__ == "__main__":
    unittest.main()
