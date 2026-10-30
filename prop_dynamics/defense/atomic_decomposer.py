import os
"""Decomposer."""

import re
import json
from typing import List

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    sent_tokenize = None

from concurrent.futures import ThreadPoolExecutor
from threading import Lock


DECOMPOSE_PROMPT = """Break down each sentence into independent atomic facts.
Every sentence must contain at least one atom.
Each atomic fact should be a simple, independent, verifiable statement.

Sentences:
{sentences}

Output format (JSON array of arrays): [["fact1 from sent1", "fact2 from sent1"], ["fact1 from sent2"], ...]
Only output the JSON, nothing else."""


def call_llm(openai_module, model: str, messages: list, 
             temperature: float = 0.1, max_tokens: int = 2000):
    try:
        resp = openai_module.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        tokens = resp.get("usage", {}).get("total_tokens", 0)
        return resp["choices"][0]["message"]["content"].strip(), tokens
    except Exception:
        print("[LLM] Request failed.")
        raise


class AtomicDecomposer:
    _executor: ThreadPoolExecutor = None
    _executor_lock = Lock()
    
    def __init__(self, model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini"), openai_module=None, word_threshold: int = 15,
                 max_workers: int = 30, enable_cache: bool = True):
        self.model = model
        self._openai = openai_module
        self.word_threshold = word_threshold
        self.tokens_used = 0
        self._tokens_lock = Lock()
        self._max_workers = max_workers
        self.enable_cache = enable_cache
        self._cache = {}
        self._ensure_nltk()
        self._init_executor()

    def _ensure_nltk(self):
        """Keep construction offline; sentence splitting has a local fallback."""
        return None
    
    def _init_executor(self):
        with AtomicDecomposer._executor_lock:
            if AtomicDecomposer._executor is None:
                AtomicDecomposer._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix="decomposer_"
                )
    
    def set_openai(self, openai_module):
        self._openai = openai_module
    
    def reset_tokens(self):
        self.tokens_used = 0
    
    def _detect_initials(self, text: str) -> List[str]:
        pattern = r"[A-Z]\. ?[A-Z]\."
        return re.findall(pattern, text)
    
    def _fix_sentence_splitter(self, sentences: List[str], initials: List[str]) -> List[str]:
        if not sentences:
            return sentences
        
        fixed = []
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            
            if len(sent.split()) <= 2 and fixed:
                fixed[-1] = fixed[-1] + " " + sent
                i += 1
                continue
            
            if sent and sent[0].islower() and fixed:
                fixed[-1] = fixed[-1] + " " + sent
                i += 1
                continue
            
            if fixed and i > 0:
                for init in initials:
                    parts = init.split()
                    if len(parts) == 2:
                        first, second = parts[0], parts[1]
                        if fixed[-1].endswith(first) and sent.startswith(second):
                            fixed[-1] = fixed[-1] + " " + sent
                            i += 1
                            continue
            
            fixed.append(sent)
            i += 1
        
        return fixed
    
    def _split_sentences(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        
        all_sentences = []
        for para in paragraphs:
            initials = self._detect_initials(para)
            
            if HAS_NLTK:
                try:
                    sents = sent_tokenize(para)
                except Exception:
                    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
            else:
                sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
            
            sents = self._fix_sentence_splitter(sents, initials)
            all_sentences.extend(sents)
        
        return [s.strip() for s in all_sentences if s.strip() and len(s.strip()) > 3]
    
    def decompose(self, message: str) -> List[str]:
        if not message or not message.strip():
            return []
        
        sentences = self._split_sentences(message)
        if not sentences:
            return [message.strip()] if message.strip() else []
        
        if self._openai is None:
            return sentences
        
        long_sents = [(i, s) for i, s in enumerate(sentences) if len(s.split()) > self.word_threshold]
        if not long_sents:
            return sentences
        
        result_map = {}
        futures = []
        
        for orig_i, sent in long_sents:
            future = AtomicDecomposer._executor.submit(self._decompose_single, sent)
            futures.append((orig_i, future))
        
        for orig_i, future in futures:
            try:
                atoms, tokens = future.result(timeout=30)
                with self._tokens_lock:
                    self.tokens_used += tokens
                if atoms:
                    result_map[orig_i] = atoms
            except Exception:
                pass
        
        all_atoms = []
        for i, s in enumerate(sentences):
            if i in result_map:
                all_atoms.extend(result_map[i])
            else:
                all_atoms.append(s)
        return all_atoms if all_atoms else sentences
    
    def _decompose_single(self, sentence: str) -> tuple:
        if self.enable_cache:
            if sentence in self._cache:
                return self._cache[sentence], 0

        try:
            msgs = [
                {"role": "system", "content": "You are a text analysis assistant."},
                {"role": "user", "content": DECOMPOSE_PROMPT.format(sentences=f"1. {sentence}")}
            ]
            content, tokens = call_llm(self._openai, self.model, msgs, 0.1, 1000)
            
            result = [sentence]
            if content.startswith('['):
                parsed = json.loads(content)
                if parsed and isinstance(parsed[0], list):
                    result = [a.strip() for a in parsed[0] if a.strip()]
                elif parsed and isinstance(parsed[0], str):
                    result = [parsed[0].strip()]
            
            if self.enable_cache:
                self._cache[sentence] = result
            
            return result, tokens
        except Exception:
            return [sentence], 0


class SimpleDecomposer:
    def __init__(self):
        pass
    
    def decompose(self, message: str) -> List[str]:
        if not message:
            return []
        
        paragraphs = [p.strip() for p in message.split("\n") if p.strip()]
        sentences = []
        
        for para in paragraphs:
            if HAS_NLTK:
                try:
                    sents = sent_tokenize(para)
                except Exception:
                    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
            else:
                sents = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para)
            sentences.extend(sents)
        
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
