import unittest

import numpy as np
import pandas as pd
import torch

from dataset import TAG_FEATURE_COLUMNS, center_crop_or_pad
from factor_data import (
    FACTOR_COLUMNS,
    FACTOR_GROUPS,
    PooledTrainingDataset,
    SOURCE_EMOTIONS,
    SOURCES,
    TARGET_COLUMNS,
    corpus_emotion_sampler,
    shuffle_targets_within_source_emotion,
)
from losses import class_aware_clap_loss, grouped_factor_loss
from prepare_crop_features import center_intervals


class FactorDesignTests(unittest.TestCase):
    def test_scope_is_three_sources_and_dynamic_factors(self):
        self.assertEqual(
            SOURCE_EMOTIONS["iemocap"],
            ("angry", "happy", "neutral", "sad", "excited", "frustrated"),
        )
        self.assertEqual(SOURCES, ("msp", "iemocap", "crema_d"))
        self.assertEqual(set(FACTOR_GROUPS), {"pitch", "energy", "rhythm"})
        forbidden = {"gender", "Duration", "equivalentSoundLevel_dBp", "EmoAct"}
        self.assertTrue(forbidden.isdisjoint(FACTOR_COLUMNS))
        self.assertEqual(len(FACTOR_COLUMNS), 11)

    def test_center_crop_matches_feature_window(self):
        audio = np.arange(10, dtype=np.float32)
        crop, mask = center_crop_or_pad(audio, num_samples=4)
        np.testing.assert_array_equal(crop, np.arange(3, 7, dtype=np.float32))
        np.testing.assert_array_equal(mask, np.ones(4, dtype=np.int64))
        padded, mask = center_crop_or_pad(np.arange(2, dtype=np.float32), 4)
        np.testing.assert_array_equal(padded, np.array([0, 1, 0, 0]))
        np.testing.assert_array_equal(mask, np.array([1, 1, 0, 0]))

    def test_feature_intervals_match_center_crop(self):
        durations = pd.Series([2.0, 5.0, 9.0])
        starts, ends = center_intervals(durations)
        np.testing.assert_allclose(starts, [0.0, 0.0, 2.0])
        np.testing.assert_allclose(ends, [2.0, 5.0, 7.0])
        np.testing.assert_allclose(ends - starts, [2.0, 5.0, 5.0])

    def test_dataset_reads_one_factor_vector(self):
        row = {"emotion": "angry", "wav_name": "unused.wav", "gender": "unknown"}
        row.update({column: 0.5 for column in TARGET_COLUMNS})
        row.update({f"{column}.quantile": 0.5 for column in TAG_FEATURE_COLUMNS})
        dataset = PooledTrainingDataset(pd.DataFrame([row]), True)
        dataset.df.loc[0, "audio_path"] = "unused.wav"
        with unittest.mock.patch("dataset.load_waveform", return_value=np.zeros(8)):
            sample = dataset[0]
        self.assertEqual(sample[-1].shape, (len(TARGET_COLUMNS),))

    def test_shuffle_preserves_joint_vectors_inside_each_stratum(self):
        rows = []
        for source in SOURCES:
            for emotion in ("angry", "sad"):
                for item in range(5):
                    row = {"_source_corpus": source, "emotion": emotion}
                    row.update(
                        {
                            column: float(item * 100 + index)
                            for index, column in enumerate(TARGET_COLUMNS)
                        }
                    )
                    rows.append(row)
        frame = pd.DataFrame(rows)
        shuffled = shuffle_targets_within_source_emotion(frame, seed=3407)
        for key, positions in frame.groupby(
            ["_source_corpus", "emotion"], sort=False
        ).indices.items():
            before = frame.loc[positions, list(TARGET_COLUMNS)].to_numpy()
            after = shuffled.loc[positions, list(TARGET_COLUMNS)].to_numpy()
            self.assertEqual(
                sorted(map(tuple, before)),
                sorted(map(tuple, after)),
                key,
            )
            self.assertFalse(
                np.any(np.all(before == after, axis=1)),
                f"fixed target row remains in stratum {key}",
            )

    def test_sampler_has_equal_source_then_emotion_mass(self):
        frame = pd.DataFrame(
            {
                "_source_corpus": ["a", "a", "a", "b", "b", "b", "b"],
                "emotion": ["x", "x", "y", "x", "y", "y", "y"],
            }
        )
        sampler = corpus_emotion_sampler(frame)
        frame["weight"] = sampler.weights.numpy()
        masses = frame.groupby(["_source_corpus", "emotion"])["weight"].sum()
        np.testing.assert_allclose(masses.to_numpy(), np.full(4, 0.5))
        source_mass = masses.groupby(level=0).sum()
        np.testing.assert_allclose(source_mass.to_numpy(), np.ones(2))

    def test_class_aware_loss_is_invariant_within_emotion(self):
        text = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]), dim=1
        )
        audio = torch.nn.functional.normalize(
            torch.tensor([[0.9, 0.1], [1.0, 0.0], [0.1, 0.9]]), dim=1
        )
        labels = ("angry", "angry", "sad")
        baseline = class_aware_clap_loss(text, audio, labels, torch.tensor(2.0))
        swapped = class_aware_clap_loss(
            text[[1, 0, 2]], audio, labels, torch.tensor(2.0)
        )
        self.assertTrue(torch.allclose(baseline, swapped))

    def test_grouped_factor_loss_prefers_correct_targets(self):
        target = torch.linspace(0, 1, len(FACTOR_COLUMNS)).unsqueeze(0)
        predictions = {}
        offset = 0
        for name, columns in FACTOR_GROUPS.items():
            width = len(columns)
            predictions[name] = target[:, offset : offset + width].clone()
            offset += width
        correct, parts = grouped_factor_loss(predictions, target)
        self.assertEqual(set(parts), set(FACTOR_GROUPS))
        wrong, _ = grouped_factor_loss(
            {name: 1.0 - value for name, value in predictions.items()}, target
        )
        self.assertEqual(correct.item(), 0.0)
        self.assertGreater(wrong.item(), correct.item())


if __name__ == "__main__":
    unittest.main()
