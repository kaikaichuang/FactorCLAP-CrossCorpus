import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from factor_data import SOURCES
from train_pooled import REPO_ROOT, experiment_contract


ROOT = Path(__file__).resolve().parents[1]


class ExperimentContractTests(unittest.TestCase):
    def test_training_never_references_test_split(self):
        source = (ROOT / "train_pooled.py").read_text()
        self.assertNotIn('"test.csv"', source)
        self.assertNotIn("/test/", source)
        self.assertIn("mean_uar", source)
        self.assertIn("dev_csvs", source)

    def test_iemocap_native_evaluation_is_six_class(self):
        runner = (ROOT / "scripts/run_case.sh").read_text()
        self.assertIn(
            "native_args=(--emotions angry happy neutral sad excited frustrated)",
            runner,
        )
        trainer = (ROOT / "train_pooled.py").read_text()
        self.assertIn("emotions=emotions_by_source[source]", trainer)

    def test_all_conditions_and_cameo_are_scheduled(self):
        runner = (ROOT / "scripts/run_case.sh").read_text()
        for condition in (
            "e0_emotion",
            "e1_smooth",
            "e2_factor",
            "e3_shuffled_factor",
        ):
            self.assertIn(condition, runner)
        for target in (
            "cafe",
            "emns",
            "emozionalmente",
            "enterface",
            "jl_corpus",
            "mesd",
            "nemo",
            "oreau",
            "pavoque",
            "ravdess",
            "resd",
            "subesco",
        ):
            self.assertIn(target, runner)
        self.assertIn("native", runner)
        self.assertIn("shared4", runner)

    def test_training_is_blocked_until_full_preflight(self):
        runner = (ROOT / "scripts/run_case.sh").read_text()
        preparation = (
            ROOT / "scripts/nchc/prepare_features.sbatch"
        ).read_text()
        self.assertIn('"$feature_root/READY"', runner)
        self.assertIn("preflight_inputs.py", preparation)

    def test_each_condition_gets_an_independent_slurm_job(self):
        submitter = (ROOT / "scripts/nchc/submit_all.sh").read_text()
        self.assertIn("scripts/nchc/train_case.sbatch", submitter)
        self.assertIn('dependency="afterok:$feature_job"', submitter)
        for condition in (
            "e0_emotion",
            "e1_smooth",
            "e2_factor",
            "e3_shuffled_factor",
        ):
            self.assertIn(condition, submitter)

    def test_factor_checkpoint_loader_is_registered(self):
        source = (ROOT / "eval_csv.py").read_text()
        self.assertIn('name.startswith("factor_heads.")', source)
        self.assertIn("FactorCLAP", source)

    def test_resume_contract_detects_prepared_feature_change(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_root = root / "features"
            feature_root.mkdir()
            initial_state = root / "initial.pth.tar"
            initial_state.write_bytes(b"shared initial state")
            train_csvs = {}
            dev_csvs = {}
            for source in SOURCES:
                train_csvs[source] = str(root / f"{source}_train.csv")
                dev_csvs[source] = str(root / f"{source}_development.csv")
                Path(train_csvs[source]).write_text("sample_id\\ntrain\\n")
                Path(dev_csvs[source]).write_text("sample_id\\ndev\\n")
                (feature_root / f"{source}_train_eGeMAPSv02.csv").write_text(
                    "sample_id,feature\\ntrain,0.1\\n"
                )
            with open(REPO_ROOT / "configs/config.yaml") as file:
                config = yaml.safe_load(file)
            args = SimpleNamespace(
                condition="e2_factor",
                factor_weight=64.0,
                feature_root=str(feature_root),
                initial_state=str(initial_state),
            )
            with patch("train_pooled.subprocess.check_output", return_value="commit\\n"):
                saved = experiment_contract(args, config, train_csvs, dev_csvs)
                (feature_root / "msp_train_eGeMAPSv02.csv").write_text(
                    "sample_id,feature\\ntrain,0.2\\n"
                )
                current = experiment_contract(args, config, train_csvs, dev_csvs)
            self.assertNotEqual(saved, current)
            self.assertNotEqual(
                saved["input_sha256"]["msp_features"],
                current["input_sha256"]["msp_features"],
            )


if __name__ == "__main__":
    unittest.main()
