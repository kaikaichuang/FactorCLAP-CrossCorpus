import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from train_expert import REPO_ROOT, experiment_contract


ROOT = Path(__file__).resolve().parents[1]


class ExperimentContractTests(unittest.TestCase):
    def test_training_selects_only_own_development_split(self):
        source = (ROOT / "train_expert.py").read_text()
        self.assertNotIn('"test.csv"', source)
        self.assertNotIn("/test/", source)
        self.assertIn('dev_uar = dev["UAR"]', source)
        self.assertNotIn("mean_uar", source)

    def test_iemocap_native_evaluation_is_six_class(self):
        runner = (ROOT / "scripts/run_case.sh").read_text()
        self.assertIn(
            "native_args=(--emotions angry happy neutral sad excited frustrated)",
            runner,
        )
        trainer = (ROOT / "train_expert.py").read_text()
        self.assertIn("emotions=emotions", trainer)

    def test_three_conditions_and_all_targets_are_scheduled(self):
        runner = (ROOT / "scripts/run_case.sh").read_text()
        for condition in (
            "e0_emotion",
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
        preparation = (ROOT / "scripts/nchc/prepare_features.sbatch").read_text()
        train_job = (ROOT / "scripts/nchc/train_case.sbatch").read_text()
        prepare_once = (ROOT / "scripts/nchc/prepare_once.sh").read_text()
        self.assertIn('"$feature_root/READY"', runner)
        self.assertIn("scripts/nchc/prepare_once.sh", preparation)
        self.assertIn("preflight_inputs.py", preparation)
        self.assertIn("#SBATCH --cpus-per-task=12", preparation)
        self.assertIn('--num-workers "$SLURM_CPUS_PER_TASK"', preparation)
        self.assertIn("conda run --no-capture-output -n clap", preparation)
        self.assertIn("conda run --no-capture-output -n clap", train_job)
        self.assertNotIn("conda activate", preparation + train_job + prepare_once)
        submitter = (ROOT / "scripts/nchc/submit_all.sh").read_text()
        self.assertNotIn("bash scripts/nchc/prepare_once.sh", submitter)

    def test_nine_source_condition_jobs_are_defined(self):
        submitter = (ROOT / "scripts/nchc/submit_all.sh").read_text()
        self.assertIn("scripts/nchc/train_case.sbatch", submitter)
        self.assertIn('dependency="afterok:$feature_job"', submitter)
        self.assertIn("sources=(msp iemocap crema_d)", submitter)
        self.assertIn(
            "conditions=(e0_emotion e2_factor e3_shuffled_factor)", submitter
        )
        runner = (ROOT / "scripts/run_case.sh").read_text()
        self.assertIn('for required_source in msp iemocap crema_d', runner)
        self.assertIn('--source "$source"', runner)

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
            train_csv = root / "msp_train.csv"
            dev_csv = root / "msp_development.csv"
            train_csv.write_text("sample_id\ntrain\n")
            dev_csv.write_text("sample_id\ndev\n")
            feature_csv = feature_root / "msp_train_eGeMAPSv02.csv"
            feature_csv.write_text("sample_id,feature\ntrain,0.1\n")
            with open(REPO_ROOT / "configs/config.yaml") as file:
                config = yaml.safe_load(file)
            args = SimpleNamespace(
                source="msp",
                condition="e2_factor",
                factor_weight=64.0,
                feature_root=str(feature_root),
                initial_state=str(initial_state),
            )
            emotions = ["angry", "happy"]
            with patch("train_expert.subprocess.check_output", return_value="commit\n"):
                saved = experiment_contract(
                    args, config, emotions, str(train_csv), str(dev_csv)
                )
                feature_csv.write_text("sample_id,feature\ntrain,0.2\n")
                current = experiment_contract(
                    args, config, emotions, str(train_csv), str(dev_csv)
                )
            self.assertNotEqual(
                saved["input_sha256"]["msp_features"],
                current["input_sha256"]["msp_features"],
            )


if __name__ == "__main__":
    unittest.main()
