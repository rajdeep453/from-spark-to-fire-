"""Small offline regression tests for repeatable experiment behavior."""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from prop_dynamics.threat.config import stable_seed
from prop_dynamics.defense import atomic_decomposer
from prop_dynamics.defense.atomic_decomposer import SimpleDecomposer


class ReproducibilityTests(unittest.TestCase):
    def test_threat_seed_is_repeatable_across_python_hash_seeds(self):
        code = "from prop_dynamics.threat.config import stable_seed; print(stable_seed(42, 'task-7', 2))"
        outputs = []
        for hash_seed in ("1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            outputs.append(
                subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip()
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(int(outputs[0]), stable_seed(42, "task-7", 2))

    def test_threat_seed_rejects_invalid_arguments(self):
        with self.assertRaises(TypeError):
            stable_seed(True, "task")
        with self.assertRaises(ValueError):
            stable_seed(42, "")
        with self.assertRaises(ValueError):
            stable_seed(42, "task", -1)

    def test_sentence_split_falls_back_locally_when_nltk_data_is_missing(self):
        with patch.object(atomic_decomposer, "HAS_NLTK", True), patch.object(
            atomic_decomposer, "sent_tokenize", side_effect=LookupError("tokenizer unavailable")
        ):
            sentences = SimpleDecomposer().decompose("First sentence. Second sentence.")
        self.assertEqual(sentences, ["First sentence.", "Second sentence."])


if __name__ == "__main__":
    unittest.main()
