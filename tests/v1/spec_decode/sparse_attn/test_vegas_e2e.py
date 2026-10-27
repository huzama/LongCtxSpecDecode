# SPDX-License-Identifier: Apache-2.0
"""Opt-in model stress test; each engine runs in a fresh process.

Set VEGAS_TEST_MODEL to a locally cached checkpoint and VEGAS_TEST_FA to 2 or 4.
Exercises chunked prefill, changing batches, short prefixes, a context boundary,
and both eager and normal graph execution against the full-attention target.
"""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


class MemoryProbe:
    def vegas_memory(self):
        runner = self.model_runner
        draft = getattr(runner, "drafter", None)
        gathered = draft.attn_overrider._draft_kv if draft is not None else None
        scratch = gathered.k.nbytes + gathered.v.nbytes if gathered is not None else 0
        return dict(model_bytes=runner.model_memory_usage, gather_bytes=scratch)


def worker(model, version, eager, ratio, output):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    checkpoint = (
        model
        if Path(model).is_dir()
        else snapshot_download(model, local_files_only=True)
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    text = "The archive contains blue, red and green records. The key is amber. " * 1024
    tokens = tokenizer.encode(text)
    lengths = [17, 257, 1023, 4085]
    prompts = [dict(prompt_token_ids=tokens[:n]) for n in lengths]
    params = [
        SamplingParams(
            temperature=0, max_tokens=min(32, 4096 - n), ignore_eos=True, logprobs=5
        )
        for n in lengths
    ]
    llm = LLM(
        model=checkpoint,
        dtype="bfloat16",
        seed=41,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.6,
        enforce_eager=eager,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_cascade_attn=True,
        async_scheduling=False,
        attention_config=dict(backend="FLASH_ATTN", flash_attn_version=version),
        worker_extension_cls=(
            "tests.v1.spec_decode.sparse_attn.test_vegas_e2e.MemoryProbe"
        ),
        compilation_config=dict(cudagraph_capture_sizes=[1, 2, 4, 8, 16, 28]),
        speculative_config=(
            dict(
                method="sparse_attn",
                sparse_attn_algorithm="vegas",
                sparse_attn_ratio=0.07 if ratio == 1 else ratio,
                sparse_attn_min_tokens=4096 if ratio == 1 else 16,
                num_speculative_tokens=6,
            )
            if ratio
            else None
        ),
    )

    try:
        stats = llm.collective_rpc("vegas_memory")
        results, probabilities = [], []
        for order in ([0, 1, 2, 3], [2], [1, 0, 2], [3, 0]):
            generated = llm.generate(
                [prompts[i] for i in order], [params[i] for i in order], use_tqdm=False
            )
            results.append([list(item.outputs[0].token_ids) for item in generated])
            probabilities.append(
                [
                    [
                        {str(token): value.logprob for token, value in step.items()}
                        for step in item.outputs[0].logprobs
                    ]
                    for item in generated
                ]
            )
        Path(output).write_text(
            json.dumps(dict(outputs=results, logprobs=probabilities, memory=stats))
        )
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    if not os.environ.get("VEGAS_TEST_MODEL"):
        pytest.skip("Set VEGAS_TEST_MODEL to a cached checkpoint")
    root = tmp_path_factory.mktemp("vegas-e2e")
    cache = {}

    def run(eager, ratio):
        key = (eager, ratio)
        if key not in cache:
            name = f"{'eager' if eager else 'graphs'}-{ratio}"
            output = root / f"{name}.json"
            log = root / f"{name}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                os.environ["VEGAS_TEST_MODEL"],
                os.environ.get("VEGAS_TEST_FA", "2"),
                str(int(eager)),
                str(ratio),
                str(output),
            ]
            repo = str(Path(__file__).resolve().parents[4])
            python_path = filter(None, [repo, os.environ.get("PYTHONPATH")])
            env = {
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": os.pathsep.join(python_path),
            }
            with log.open("w") as stream:
                result = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                    env=env,
                )
            assert result.returncode == 0, f"{log}\n{log.read_text()[-16000:]}"
            cache[key] = json.loads(output.read_text())
        return cache[key]

    return run


@pytest.mark.parametrize("eager", [True, False])
@pytest.mark.parametrize("ratio", [0.07, 1.0])
def test_greedy_verification_and_batch_lifecycle(engines, eager, ratio):
    dense = engines(eager, 0.0)
    vegas = engines(eager, ratio)
    assert len(dense["outputs"]) == len(vegas["outputs"]) == 4
    # Every emitted token must maximize its actual target-verifier distribution.
    for result in (dense, vegas):
        assert len(result["outputs"]) == len(result["logprobs"])
        for rows, probabilities in zip(result["outputs"], result["logprobs"]):
            assert len(rows) == len(probabilities)
            for tokens, steps in zip(rows, probabilities):
                assert len(tokens) == len(steps)
                for token, scores in zip(tokens, steps):
                    assert scores[str(token)] == max(scores.values())
    for round_id, (expected, actual) in enumerate(
        zip(dense["outputs"], vegas["outputs"])
    ):
        assert len(expected) == len(actual)
        for request, (left, right) in enumerate(zip(expected, actual)):
            assert len(left) == len(right)
            for position, (target, token) in enumerate(zip(left, right)):
                if target != token:
                    scores = dense["logprobs"][round_id][request][position]
                    assert scores.get(str(token)) == scores[str(target)], (
                        f"Non-tied divergence at {round_id=}, {request=}, {position=}: "
                        f"{target=} {token=} {scores=}"
                    )
                    warnings.warn(
                        f"Exact dense-target tie at round {round_id}, "
                        f"request {request}, token {position}; later prefixes differ",
                        stacklevel=2,
                    )
                    break
    assert len(dense["memory"]) == len(vegas["memory"])
    for base, stats in zip(dense["memory"], vegas["memory"]):
        assert stats["gather_bytes"] > 0
        assert stats["model_bytes"] - base["model_bytes"] >= stats["gather_bytes"]


if __name__ == "__main__":
    worker(
        sys.argv[1],
        int(sys.argv[2]),
        bool(int(sys.argv[3])),
        float(sys.argv[4]),
        sys.argv[5],
    )
