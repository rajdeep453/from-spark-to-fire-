"""Tests for the GUARD result explanation contract."""

import sqlite3
import tempfile
import unittest
import importlib.util
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from prop_dynamics.guard.explain import explain_claim, explain_result
from prop_dynamics.guard.fact_store import FactStore
from prop_dynamics.defense.genealogy_graph import GenealogyGraph, VerificationStatus
from prop_dynamics.defense.classifier import ClassifyResult, ClassificationOutput, Classifier
from prop_dynamics.defense.classifier_nli import NLIClassifier
from prop_dynamics.defense.config import DefenseConfig
from prop_dynamics.defense.middleware import AtomColor, AtomInfo, DefenseMiddleware, ProcessResult
from prop_dynamics.defense.verifier import ExternalVerifier, VerificationOutput, VerifyResult

if importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx") is not None:
    from fastapi.testclient import TestClient
    from prop_dynamics.guard.app import create_app
else:
    TestClient = None
    create_app = None


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


class ExplainClaimTests(unittest.TestCase):
    class Classifier:
        def __init__(self, label="entailment", confidence=0.9, evidence=None):
            self.label = label
            self.confidence = confidence
            self.evidence = ["A matching fact."] if evidence is None else evidence

        def classify(self, claim, graph, top_k=3):
            return SimpleNamespace(
                result=SimpleNamespace(value=self.label),
                confidence=self.confidence,
                reason="classifier reason",
                evidence=self.evidence,
            )

    def test_uses_only_classifier_decision_and_evidence(self):
        result = explain_claim("Claim", object(), self.Classifier())

        self.assertEqual(result["decision"], "supported")
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["evidence"], [{"content": "A matching fact."}])
        self.assertIn("A matching fact.", result["explanation"])

    def test_maps_contradiction_and_neutral_labels(self):
        contradiction = explain_claim("Claim", object(), self.Classifier("contradiction"))
        neutral = explain_claim("Claim", object(), self.Classifier("neutral"))

        self.assertEqual(contradiction["decision"], "unsupported")
        self.assertEqual(neutral["decision"], "unknown")

    def test_empty_classifier_evidence_cannot_be_reported_as_supported(self):
        result = explain_claim("Claim", object(), self.Classifier(evidence=[]))

        self.assertEqual(result["decision"], "unknown")
        self.assertEqual(result["evidence"], [])

    def test_rejects_malformed_claim_arguments_and_classifier_results(self):
        classifier = self.Classifier()
        for claim, error in [(None, TypeError), (" ", ValueError)]:
            with self.subTest(claim=claim), self.assertRaises(error):
                explain_claim(claim, object(), classifier)
        with self.assertRaises(ValueError):
            explain_claim("Claim", object(), classifier, top_k=0)
        with self.assertRaises(TypeError):
            explain_claim("Claim", object(), object())
        with self.assertRaises(ValueError):
            explain_claim("Claim", object(), self.Classifier("maybe"))
        with self.assertRaises(ValueError):
            explain_claim("Claim", object(), self.Classifier(confidence=1.2))


class FactStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "nested" / "facts.sqlite3"
        self.graph = GenealogyGraph()
        self.store = FactStore(self.db_path, self.graph)

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_add_persists_provenance_and_indexes_verified_fact(self):
        row = self.store.add("  Water   freezes at 0 C. ", source="  lab notes ")

        self.assertTrue(row["created"])
        self.assertEqual(row["content"], "Water freezes at 0 C.")
        self.assertEqual(row["source"], "lab notes")
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(len(self.graph), 1)
        self.assertEqual(self.graph.get_verified_facts()[0].content, row["content"])

    def test_duplicate_add_is_idempotent(self):
        first = self.store.add("A fact.", source="dataset")
        duplicate = self.store.add("A   fact.", source="another source")

        self.assertTrue(first["created"])
        self.assertFalse(duplicate["created"])
        self.assertEqual(first["id"], duplicate["id"])
        self.assertEqual(duplicate["source"], "dataset")
        self.assertEqual(len(self.graph), 1)

    def test_batch_insert_is_atomic_and_duplicates_within_batch_are_deduplicated(self):
        rows = self.store.add_many(["one", "two", "one"], source="batch")

        self.assertEqual([row["created"] for row in rows], [True, True, False])
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(len(self.graph), 2)

    def test_batch_rolls_back_sqlite_and_graph_if_indexing_fails(self):
        original_add = self.graph.add

        def fail_on_second_fact(content, *args, **kwargs):
            if content == "two":
                raise RuntimeError("injected indexing failure")
            return original_add(content, *args, **kwargs)

        self.graph.add = fail_on_second_fact
        with self.assertRaises(RuntimeError):
            self.store.add_many(["one", "two"])
        self.graph.add = original_add

        self.assertEqual(self.store.count(), 0)
        self.assertEqual(len(self.graph), 0)

    def test_invalid_fact_inputs_are_rejected(self):
        for value, error in [(None, TypeError), (0, TypeError), ("  ", ValueError)]:
            with self.subTest(value=value), self.assertRaises(error):
                self.store.add(value)
        with self.assertRaises(ValueError):
            self.store.add("x" * 8001)
        with self.assertRaises(ValueError):
            self.store.add("valid", source=" ")

    def test_fact_listing_is_paginated_and_stable(self):
        for content in ("one", "two", "three"):
            self.store.add(content)

        self.assertEqual([row["content"] for row in self.store.list(limit=2, offset=1)], ["two", "three"])
        with self.assertRaises(ValueError):
            self.store.list(limit=0)

    def test_delete_removes_fact_from_database_and_search_graph(self):
        row = self.store.add("A verified fact.")

        self.assertTrue(self.store.delete(row["id"]))
        self.assertFalse(self.store.delete(row["id"]))
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(len(self.graph), 0)
        with self.assertRaises(ValueError):
            self.store.delete(0)

    def test_facts_survive_close_and_reopen(self):
        row = self.store.add("Persisted fact.")
        self.store.close()
        self.store = FactStore(self.db_path, self.graph)

        self.assertEqual(self.store.list()[0]["id"], row["id"])
        self.assertEqual(self.graph.get_verified_facts()[0].content, "Persisted fact.")

    def test_migrates_existing_database_without_source_column(self):
        self.store.close()
        legacy_path = self.db_path.with_name("legacy.sqlite3")
        conn = sqlite3.connect(legacy_path)
        try:
            conn.execute(
                "CREATE TABLE facts (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "content TEXT NOT NULL UNIQUE, created_at REAL NOT NULL)"
            )
            conn.execute("INSERT INTO facts (content, created_at) VALUES (?, ?)", ("legacy", 1.0))
            conn.commit()
        finally:
            conn.close()
        self.store = FactStore(legacy_path, self.graph)

        self.assertEqual(self.store.list()[0]["source"], "manual")
        self.assertEqual(self.graph.get_verified_facts()[0].content, "legacy")

    def test_closed_store_rejects_operations_and_close_is_idempotent(self):
        self.store.close()
        self.store.close()
        with self.assertRaises(RuntimeError):
            self.store.count()

    def test_expands_home_directory_in_database_path(self):
        with patch("pathlib.Path.expanduser", return_value=Path(self.temp_dir.name) / "expanded.sqlite3"):
            path = Path("~/expanded.sqlite3")
            store = FactStore(path, GenealogyGraph())
        try:
            self.assertTrue((Path(self.temp_dir.name) / "expanded.sqlite3").exists())
        finally:
            store.close()


class DefenseConfigTests(unittest.TestCase):
    def test_model_environment_settings_are_read_when_config_is_created(self):
        with patch.dict("os.environ", {
            "GUARD_NLI_MODEL": "local/nli-test",
            "GUARD_EMBEDDING_MODEL": "local/embedding-test",
        }):
            config = DefenseConfig()

        self.assertEqual(config.nli_model, "local/nli-test")
        self.assertEqual(config.embedding_model, "local/embedding-test")


class GenealogyGraphTests(unittest.TestCase):
    def test_fallback_embedding_is_stable_and_does_not_reset_numpy_rng(self):
        import numpy as np

        graph = GenealogyGraph()
        np.random.seed(42)
        expected_next = np.random.random()
        np.random.seed(42)
        first = graph._get_embedding("same fact")
        observed_next = np.random.random()
        second = GenealogyGraph()._get_embedding("same fact")

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(expected_next, observed_next)

    def test_search_excludes_unverified_and_rejected_facts(self):
        graph = GenealogyGraph(similarity_threshold=-1.0)
        verified = graph.add("trusted fact", "source", VerificationStatus.VERIFIED)
        unverified = graph.add("trusted fact", "agent", VerificationStatus.UNVERIFIED)
        rejected = graph.add("trusted fact", "agent", VerificationStatus.REJECTED)

        results = graph.search("trusted fact", top_k=10)
        self.assertEqual([fact.id for fact, _ in results], [verified.id])
        self.assertNotIn(unverified.id, [fact.id for fact, _ in results])
        self.assertNotIn(rejected.id, [fact.id for fact, _ in results])

    def test_descendant_walk_handles_cyclic_sources(self):
        graph = GenealogyGraph()
        root = graph.add("root", "a")
        child = graph.add("child", "b", sources=[root.id])
        graph.nodes[root.id].sources = [child.id]

        self.assertEqual(set(graph.get_descendants(root.id)), {child.id})

    def test_graph_rejects_invalid_sources_and_nonfinite_query_embeddings(self):
        graph = GenealogyGraph()
        with self.assertRaises(ValueError):
            graph.add("fact", "agent", sources=[None])
        with self.assertRaises(ValueError):
            graph.add("fact", "agent", hop_count=-1)

        graph.add("trusted fact", "operator", VerificationStatus.VERIFIED)
        graph.set_embedding_func(lambda text: [float("nan")] * graph.embedding_dim)
        with self.assertRaises(ValueError):
            graph.search("claim")


class GenealogyGovernanceIntegrationTests(unittest.TestCase):
    def test_middleware_records_claim_support_edges_for_confirmed_claims(self):
        graph = GenealogyGraph()
        root = graph.add("trusted fact", "operator", VerificationStatus.VERIFIED)

        class FixedClassifier:
            def classify(self, claim, graph, top_k=5):
                return ClassificationOutput(
                    ClassifyResult.GREEN, 0.95, ["trusted fact"], "entailment", [root.id]
                )

        config = DefenseConfig(
            classifier_type="simple", use_local_embedding=False,
            enable_decomposer=False, enable_verifier=False,
        )
        middleware = DefenseMiddleware(config, graph=graph)
        middleware.classifier = FixedClassifier()

        result = middleware.process_message("Analyst", "Reviewer", "downstream claim")
        child = next(fact for fact in graph.nodes.values() if fact.content == "downstream claim")

        self.assertTrue(result.can_send)
        self.assertEqual(child.sources, [root.id])
        self.assertEqual(graph.get_descendants(root.id), [child.id])

        graph.prune_branch(root.id)
        self.assertEqual(child.status, VerificationStatus.REJECTED)

    def test_uncertain_claims_are_recorded_but_never_trusted(self):
        graph = GenealogyGraph(similarity_threshold=-1.0)
        root = graph.add("trusted fact", "operator", VerificationStatus.VERIFIED)

        class FixedClassifier:
            def classify(self, claim, graph, top_k=5):
                return ClassificationOutput(
                    ClassifyResult.YELLOW, 0.5, ["trusted fact"], "neutral", [root.id]
                )

        config = DefenseConfig(
            classifier_type="simple", use_local_embedding=False,
            enable_decomposer=False, enable_verifier=False,
        )
        middleware = DefenseMiddleware(config, graph=graph)
        middleware.classifier = FixedClassifier()

        result = middleware.process_message("Analyst", "Reviewer", "uncertain claim")
        child = next(fact for fact in graph.nodes.values() if fact.content == "uncertain claim")

        self.assertTrue(result.can_send)
        self.assertEqual(child.status, VerificationStatus.UNVERIFIED)
        self.assertEqual(child.sources, [root.id])
        self.assertNotIn(child.id, [fact.id for fact, _ in graph.search(child.content, top_k=10)])

    def test_verified_claim_keeps_the_verifiers_actual_evidence(self):
        graph = GenealogyGraph()
        root = graph.add("trusted fact", "operator", VerificationStatus.VERIFIED)

        class FixedClassifier:
            def classify(self, claim, graph, top_k=5):
                return ClassificationOutput(
                    ClassifyResult.YELLOW, 0.5, [], "new claim", []
                )

        class FixedVerifier:
            def verify(self, claim):
                return VerificationOutput(
                    VerifyResult.SUPPORTED, ["verified evidence"],
                    [f"lineage:{root.id}"], 0.95, "supported",
                )

        config = DefenseConfig(
            strategy="strict", classifier_type="simple", use_local_embedding=False,
            enable_decomposer=False, enable_verifier=True,
        )
        middleware = DefenseMiddleware(config, graph=graph)
        middleware.classifier = FixedClassifier()
        middleware.verifier = FixedVerifier()

        result = middleware.process_message("Analyst", "Reviewer", "verified claim")
        child = next(fact for fact in graph.nodes.values() if fact.content == "verified claim")

        self.assertTrue(result.can_send)
        self.assertEqual(child.status, VerificationStatus.VERIFIED)
        self.assertEqual(child.sources, [root.id])
        self.assertEqual(result.send_buffer[0].evidence, ["verified evidence"])


class ClassifierProvenanceTests(unittest.TestCase):
    class FakeOpenAI:
        class ChatCompletion:
            response = '["ENTAILMENT", "CONTRADICTION"]'
            last_messages = None

            @classmethod
            def create(cls, **kwargs):
                cls.last_messages = kwargs["messages"]
                return {
                    "choices": [{"message": {"content": cls.response}}],
                    "usage": {"total_tokens": 1},
                }

    class FakeGraph:
        def __init__(self):
            self.facts = {
                "claim one": [SimpleNamespace(id="fact-one", content="Evidence for one")],
                "claim two": [SimpleNamespace(id="fact-two", content="Evidence for two")],
                "claim none": [],
            }

        def search(self, query, top_k=3):
            return [(fact, 0.9) for fact in self.facts[query]]

    def test_batch_evidence_and_lineage_are_specific_to_each_claim(self):
        classifier = Classifier(openai_module=self.FakeOpenAI)
        outputs = classifier.classify_batch(
            ["claim one", "claim two", "claim none"], self.FakeGraph()
        )

        self.assertEqual([output.result for output in outputs], [
            ClassifyResult.GREEN, ClassifyResult.RED, ClassifyResult.YELLOW,
        ])
        self.assertEqual(outputs[0].evidence_ids, ["fact-one"])
        self.assertEqual(outputs[1].evidence_ids, ["fact-two"])
        self.assertEqual(outputs[2].evidence_ids, [])
        prompt = self.FakeOpenAI.ChatCompletion.last_messages[1]["content"]
        self.assertIn("consider only these facts", prompt)
        self.assertNotIn("Evidence for two", prompt.split("Claim 1:", 1)[1].split("Claim 2:", 1)[0])

    def test_non_exact_batch_labels_are_neutral(self):
        self.FakeOpenAI.ChatCompletion.response = '["NOT ENTAILMENT", "CONTRADICTION"]'
        classifier = Classifier(openai_module=self.FakeOpenAI)
        outputs = classifier.classify_batch(["claim one", "claim two"], self.FakeGraph())

        self.assertEqual(outputs[0].result, ClassifyResult.YELLOW)
        self.assertEqual(outputs[1].result, ClassifyResult.RED)

    def test_nli_confidence_is_normalized_from_cross_encoder_logits(self):
        classifier = object.__new__(NLIClassifier)
        classifier.labels = ["contradiction", "entailment", "neutral"]
        classifier.tokens_used = 0
        classifier.model = SimpleNamespace(predict=lambda pairs: [[0.0, 2.0, -1.0]])
        graph = SimpleNamespace(search=lambda claim, top_k: [
            (SimpleNamespace(id="fact", content="trusted premise"), 0.9)
        ])

        output = classifier.classify("claim", graph)

        self.assertEqual(output.result, ClassifyResult.GREEN)
        self.assertGreater(output.confidence, 0.0)
        self.assertLessEqual(output.confidence, 1.0)
        self.assertAlmostEqual(output.confidence, 0.8437947, places=6)

    def test_verifier_does_not_treat_not_supported_as_supported(self):
        class FakeVerifierClient:
            class ChatCompletion:
                @staticmethod
                def create(**kwargs):
                    return {
                        "choices": [{"message": {"content": "NOT SUPPORTED"}}],
                        "usage": {"total_tokens": 1},
                    }

        verifier = ExternalVerifier(openai_module=FakeVerifierClient)
        verifier.set_external_search_func(lambda query: [("evidence", "https://example.test")])

        self.assertEqual(verifier.verify("claim").result, VerifyResult.NOT_ENOUGH_INFO)

    def test_verifier_cache_is_invalidated_when_trusted_graph_changes(self):
        class FakeVerifierClient:
            class ChatCompletion:
                response = "SUPPORTED"

                @classmethod
                def create(cls, **kwargs):
                    return {
                        "choices": [{"message": {"content": cls.response}}],
                        "usage": {"total_tokens": 1},
                    }

        graph = GenealogyGraph(similarity_threshold=-1.0)
        graph.add("initial evidence", "operator", VerificationStatus.VERIFIED)
        verifier = ExternalVerifier(openai_module=FakeVerifierClient)
        verifier.set_knowledge_base(graph)

        first = verifier.verify("claim")
        FakeVerifierClient.ChatCompletion.response = "CONTRADICTED"
        graph.add("new evidence", "operator", VerificationStatus.VERIFIED)
        second = verifier.verify("claim")

        self.assertEqual(first.result, VerifyResult.SUPPORTED)
        self.assertEqual(second.result, VerifyResult.CONTRADICTED)


class _FakeClassifier:
    def classify(self, claim, graph, top_k=3):
        return SimpleNamespace(
            result=SimpleNamespace(value="entailment"),
            confidence=0.93,
            reason="NLI: entailment",
            evidence=["A matching trusted fact."],
        )


class _FakeGuard:
    def process_message(self, sender, receiver, message):
        atom = AtomInfo(content=message, color=AtomColor.GREEN, reason="matches", evidence=["fact"])
        return ProcessResult(True, [atom], [], None, message)


@unittest.skipIf(TestClient is None, "FastAPI/httpx test dependencies are not installed")
class GuardApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "api.sqlite3"
        self.app = create_app(
            db_path=str(self.db_path),
            graph=GenealogyGraph(),
            nli_model=_FakeClassifier(),
            guard_factory=lambda graph: _FakeGuard(),
            api_key="unit-test-api-key-0123456789abcdef",
        )
        self.db_existed_before_startup = self.db_path.exists()
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.headers = {"Authorization": "Bearer unit-test-api-key-0123456789abcdef"}

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.temp_dir.cleanup()

    def test_api_factory_does_not_create_database_before_startup(self):
        self.assertFalse(self.db_existed_before_startup)
        self.assertTrue(self.db_path.exists())
        self.assertEqual(self.client.get("/health").json()["status"], "ok")

    def test_health_is_available_without_api_key(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertNotIn("trusted_facts", response.json())

    def test_mutation_endpoints_require_bearer_authentication(self):
        response = self.client.post("/facts", json={"facts": ["trusted"]})
        self.assertEqual(response.status_code, 401)
        wrong_key = self.client.get("/facts", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(wrong_key.status_code, 401)

    def test_fact_crud_records_source_and_paginates(self):
        added = self.client.post(
            "/facts", headers=self.headers,
            json={"facts": ["first fact", "second fact"], "source": "unit test"},
        )
        self.assertEqual(added.status_code, 200)
        self.assertEqual(added.json()["total"], 2)
        self.assertEqual(added.json()["added"][0]["source"], "unit test")

        listed = self.client.get("/facts?limit=1&offset=1", headers=self.headers)
        self.assertEqual(listed.json()["facts"][0]["content"], "second fact")
        deleted = self.client.delete("/facts/1", headers=self.headers)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.delete("/facts/999", headers=self.headers).status_code, 404)

    def test_explain_uses_classifier_decision_and_evidence(self):
        response = self.client.post(
            "/explain", headers=self.headers, json={"claim": "A claim."}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["decision"], "supported")
        self.assertEqual(response.json()["evidence"][0]["content"], "A matching trusted fact.")

    def test_check_returns_middleware_decision_without_leaking_errors(self):
        response = self.client.post(
            "/check", headers=self.headers, json={"message": "A message."}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["can_send"])
        self.assertEqual(response.json()["claims"][0]["color"], "green")

    def test_invalid_payloads_and_oversized_bodies_are_rejected(self):
        invalid = self.client.post(
            "/facts", headers=self.headers, json={"facts": [" "]}
        )
        self.assertEqual(invalid.status_code, 422)
        oversized = self.client.post(
            "/facts", headers=self.headers,
            content=b" " * (4 * 1024 * 1024 + 1),
        )
        self.assertEqual(oversized.status_code, 413)

    def test_unconfigured_api_key_fails_closed(self):
        app = create_app(
            db_path=str(self.db_path.with_name("no-key.sqlite3")),
            graph=GenealogyGraph(), nli_model=_FakeClassifier(), api_key="",
        )
        with TestClient(app) as client:
            response = client.get("/facts", headers=self.headers)
        self.assertEqual(response.status_code, 503)

    def test_check_errors_are_generic_and_do_not_expose_exception_text(self):
        def fail_guard(graph):
            raise RuntimeError("sensitive-provider-detail")

        app = create_app(
            db_path=str(self.db_path.with_name("failure.sqlite3")),
            graph=GenealogyGraph(), nli_model=_FakeClassifier(),
            guard_factory=fail_guard,
            api_key="unit-test-api-key-0123456789abcdef",
        )
        with TestClient(app) as client:
            response = client.post(
                "/check", headers=self.headers, json={"message": "A message."}
            )
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("sensitive-provider-detail", response.text)

    def test_blocked_check_does_not_return_forwardable_partial_message(self):
        blocked_atom = AtomInfo(
            content="rejected claim", color=AtomColor.RED,
            reason="conflicts", evidence=["trusted fact"],
        )
        blocked_guard = lambda graph: SimpleNamespace(process_message=lambda *args: ProcessResult(
            False, [], [blocked_atom], "rewrite this claim", "partially assembled text",
        ))
        app = create_app(
            db_path=str(self.db_path.with_name("blocked.sqlite3")),
            graph=GenealogyGraph(), nli_model=_FakeClassifier(),
            guard_factory=blocked_guard,
            api_key="unit-test-api-key-0123456789abcdef",
        )
        with TestClient(app) as client:
            response = client.post(
                "/check", headers=self.headers, json={"message": "A message."}
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["can_send"])
        self.assertIsNone(response.json()["final_message"])


if __name__ == "__main__":
    unittest.main()
