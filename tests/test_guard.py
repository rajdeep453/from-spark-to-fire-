"""Tests for the GUARD result explanation contract."""

import unittest

from prop_dynamics.guard.explain import explain_result


class ExplainResultTests(unittest.TestCase):
    def test_explains_decision_reason_and_supplied_fact_provenance(self):
        result = {
            "decision": "supported",
            "reason": "An exact fact was found.",
            "facts": [
                {"id": "fact-1", "content": "The sky appears blue.", "source": "observation"}
            ],
        }

        self.assertEqual(
            explain_result(result),
            "GUARD decision: supported. Reason: An exact fact was found. "
            "Evidence supplied by GUARD:\n"
            "1. The sky appears blue. (id: fact-1; source: observation)",
        )

    def test_reports_empty_evidence_without_inventing_a_fact(self):
        self.assertEqual(
            explain_result({"decision": "unknown", "reason": "No match.", "facts": []}),
            "GUARD decision: unknown. Reason: No match. GUARD supplied no matching facts.",
        )

    def test_accepts_all_documented_decisions_and_optional_fields(self):
        supported = explain_result({"decision": "supported", "reason": "Evidence matched."})
        unknown = explain_result({"decision": "unknown", "reason": "No evidence."})
        unsupported = explain_result(
            {
                "decision": "unsupported",
                "reason": "The supplied fact contradicts the claim.",
                "facts": [{"content": "The observed value was 4."}],
            }
        )

        self.assertIn("decision: supported", supported)
        self.assertIn("no matching facts", supported)
        self.assertIn("decision: unknown", unknown)
        self.assertIn("decision: unsupported", unsupported)
        self.assertIn("1. The observed value was 4.", unsupported)

    def test_normalizes_decision_case_and_ignores_unrecognized_fields(self):
        result = {
            "decision": " UNSUPPORTED ",
            "reason": "A contradictory fact was supplied.",
            "facts": [{"content": "The measured result differs."}],
            "internal_debug": "not part of the explanation contract",
        }
        self.assertTrue(explain_result(result).startswith("GUARD decision: unsupported."))
        self.assertNotIn("internal_debug", explain_result(result))

    def test_rejects_invalid_result_and_fact_shapes(self):
        invalid_results = [
            (None, TypeError),
            ({"decision": None, "reason": "unclear"}, TypeError),
            ({"decision": "maybe", "reason": "unclear"}, ValueError),
            ({"decision": "supported"}, TypeError),
            ({"decision": "supported", "reason": None}, TypeError),
            ({"decision": "supported", "reason": " "}, ValueError),
            ({"decision": "supported", "reason": "ok", "facts": "not facts"}, TypeError),
            ({"decision": "supported", "reason": "ok", "facts": {"content": "x"}}, TypeError),
            ({"decision": "supported", "reason": "ok", "facts": [None]}, TypeError),
            ({"decision": "supported", "reason": "ok", "facts": [{}]}, ValueError),
            ({"decision": "supported", "reason": "ok", "facts": [{"content": None}]}, ValueError),
            ({"decision": "supported", "reason": "ok", "facts": [{"content": " "}]}, ValueError),
            ({"decision": "supported", "reason": "ok", "facts": [{"content": "fact", "id": 7}]}, TypeError),
            ({"decision": "supported", "reason": "ok", "facts": [{"content": "fact", "source": 7}]}, TypeError),
        ]
        for result, error in invalid_results:
            with self.subTest(result=result), self.assertRaises(error):
                explain_result(result)


if __name__ == "__main__":
    unittest.main()
