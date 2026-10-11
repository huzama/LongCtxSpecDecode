"""Replay one saved verification round and test rejection against clean KV.

Diagnostic only. No runtime source or model weights are modified. The worker
restores every touched cache page and request tensor before generation resumes.
Use the existing 4B/128K batch, --mode highlight --draft-select round,
--reference <saved dense tokens>, and the original --gen 512 RoPE budget.
"""

import dataclasses
import json
import os
import traceback
from pathlib import Path

import torch

from benchmarks import decode
from spotattention.serve import stats
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context


def top(logits):
    values, ids = logits.float().topk(5)
    return {"argmax": int(logits.argmax()), "top": dict(zip(ids.tolist(), values.tolist())),
            "8453_minus_9489": float(logits[8453] - logits[9489])}


class Replay:
    def __init__(self, runner, state):
        self.r = runner
        self.state = state
        self.batch = state.input_batch
        self.n = self.batch.num_reqs
        assert self.n == 6 and (self.batch.num_scheduled_tokens == 7).all()
        assert self.batch.num_tokens_after_padding == 42
        self.ids = self.batch.input_ids.clone().view(self.n, 7)
        self.starts = self.batch.positions.view(self.n, 7)[:, 0].clone()
        self.layers = runner.vllm_config.compilation_config.static_forward_context
        self.caches = []
        for name, md in state.attn_metadata.items():
            cache = self.layers[name].kv_cache
            assert cache.ndim in (3, 4), (name, cache.shape)
            assert cache.shape[2 if cache.ndim == 4 else 1] == 16
            positions = self.starts[:, None] + torch.arange(8, device=self.starts.device)
            blocks = md.block_table.gather(1, (positions // 16).long()).unique().long()
            assert int(blocks.max()) < cache.shape[0]
            self.caches.append((name, cache, blocks))
        self.original = self.snapshot()

    def snapshot(self):
        return [cache.index_select(0, blocks).clone() for _, cache, blocks in self.caches]

    def restore(self, values):
        for (_, cache, blocks), value in zip(self.caches, values):
            cache.index_copy_(0, blocks, value)

    def forward(self, ids, starts=None):
        starts = self.starts if starts is None else starts
        n, q = ids.shape
        positions = starts[:, None] + torch.arange(q, device=starts.device)
        seq_lens = (starts + q).to(torch.int32)
        qsl = torch.arange(n + 1, device=starts.device, dtype=torch.int32) * q
        metadata, slots = {}, {}
        for name, original in self.state.attn_metadata.items():
            table = original.block_table
            mapping = (table.gather(1, (positions // 16).long()).long() * 16 + positions % 16).flatten()
            # FULL graph capture uses the model limit, not the current length.
            kw = dict(num_actual_tokens=n * q, max_query_len=q,
                      max_seq_len=self.r.max_model_len, query_start_loc=qsl,
                      seq_lens=seq_lens, slot_mapping=mapping)
            if hasattr(original, "scheduler_metadata"):
                kw.update(scheduler_metadata=None, prefix_scheduler_metadata=None)
            metadata[name] = dataclasses.replace(original, **kw)
            slots[name] = mapping
        with set_forward_context(metadata, self.r.vllm_config, num_tokens=n * q,
                                 cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                 batch_descriptor=BatchDescriptor(num_tokens=n * q),
                                 slot_mapping=slots, skip_compiled=False,
                                 is_padding=torch.zeros(n * q, dtype=torch.bool, device=starts.device)):
            hidden = self.r.model(input_ids=ids.flatten().contiguous(),
                                  positions=positions.flatten().contiguous())
        logits = self.r.model.compute_logits(hidden)
        return hidden, logits.view(n, q, -1)

    def token_cache(self, cache, name, row, position):
        table = self.state.attn_metadata[name].block_table
        block = int(table[row, position // 16])
        offset = position % 16
        return cache[block, :, offset] if cache.ndim == 4 else cache[block, offset]

    def valid_prefix(self, row, length):
        start = int(self.starts[row])
        return [torch.stack([self.token_cache(cache, name, row, start + i).clone()
                             for i in range(length)]) for name, cache, _ in self.caches]

    def poison_rejected(self, row, first):
        start = int(self.starts[row])
        for name, cache, _ in self.caches:
            for i in range(first, 7):
                self.token_cache(cache, name, row, start + i).fill_(10000)


class SanityWorker(stats.SelectionStatsWorker):
    def sanity_arm(self, out, request_id, reference):
        runner = self.model_runner
        original = runner.sample_tokens
        self.sanity_checked = False

        def sample_tokens(grammar_output):
            state = runner.execute_model_state
            if not self.sanity_checked and state is not None:
                b = state.input_batch
                matches = [i for i, value in enumerate(b.req_ids)
                           if value == request_id or value.startswith(request_id + "-")]
                if matches and b.num_draft_tokens:
                    row = matches[0]
                    lo, hi = b.query_start_loc_np[row:row + 2]
                    positions = b.positions[lo:hi].tolist()
                    wanted = 130560 + 69
                    if wanted in positions:
                        offset = positions.index(wanted)
                        actual = b.input_ids[lo:lo + offset + 1].tolist()
                        expected = reference[positions[0] - 130560:70]
                        if actual == expected:
                            self.sanity_checked = True
                            path = Path(out) / "checks.json"
                            try:
                                report = self._checks(state, row, offset)
                                path.write_text(json.dumps(report, indent=2) + "\n")
                            except Exception:
                                path.write_text(json.dumps({"error": traceback.format_exc()}, indent=2) + "\n")
                                raise
            return original(grammar_output)

        runner.sample_tokens = sample_tokens
        return {"armed": True, "request_id": request_id}

    @torch.inference_mode()
    def _checks(self, state, row, offset):
        runner = self.model_runner
        replay = Replay(runner, state)
        selection = runner.model.model.sel
        active = selection.sparse_active
        original_hidden = state.hidden_states.clone()
        req = runner.req_states
        mutable = [req.num_computed_tokens.gpu, req.total_len.gpu,
                   req.last_sampled_tokens, req.all_token_ids.gpu]
        counts = runner.sampler.penalties_state.output_bin_counts
        if counts is not None:
            mutable.append(counts)
        saved = [tensor.clone() for tensor in mutable]

        def restore_requests():
            for tensor, value in zip(mutable, saved):
                tensor.copy_(value)

        report = {"request_id": state.input_batch.req_ids[row], "output_position": 70,
                  "round_input_start": int(replay.starts[row]), "offset": offset,
                  "cache_layers": len(replay.caches),
                  "replay": {"compiled": True, "max_seq_len": runner.max_model_len}}
        try:
            selection.sparse_active = False
            original_logits = runner.model.compute_logits(original_hidden).view(6, 7, -1)
            clean_hidden, clean_logits = replay.forward(replay.ids)
            report["verifier"] = {
                "hidden_bitwise_equal": torch.equal(original_hidden, clean_hidden),
                "logits_bitwise_equal": torch.equal(original_logits, clean_logits),
                "max_hidden_difference": float((original_hidden.float() - clean_hidden.float()).abs().max()),
                "max_logit_difference": float((original_logits - clean_logits).abs().max()),
                "live": top(original_logits[row, offset]), "clean": top(clean_logits[row, offset]),
            }
            replay.restore(replay.original)
            sequential = []
            for i in range(7):
                _, logits = replay.forward(replay.ids[:, i:i + 1], replay.starts + i)
                sequential.append(logits)
            sequential = torch.cat(sequential, dim=1)
            report["shape_control"] = {
                "sequential": top(sequential[row, offset]), "chunk": top(clean_logits[row, offset]),
                "max_logit_difference": float((sequential - clean_logits).abs().max()),
                "argmax_differences": int((sequential.argmax(-1) != clean_logits.argmax(-1)).sum()),
            }

            # Construct a fully accepted proposal under a fixed dense chunk shape.
            # Causality makes this converge in at most six prefix corrections.
            canonical = replay.ids.clone()
            for iteration in range(8):
                replay.restore(replay.original)
                _, logits = replay.forward(canonical)
                predicted = logits.argmax(-1)
                if torch.equal(canonical[:, 1:], predicted[:, :-1]):
                    break
                canonical[:, 1:] = predicted[:, :-1]
            else:
                raise AssertionError("dense chunk proposal did not converge")
            prefix = replay.valid_prefix(row, 7)
            report["canonical_iterations"] = iteration + 1
            report["rejections"] = []
            for reject_at in (0, 2, 5):
                replay.restore(replay.original)
                restore_requests()
                candidate = canonical.clone()
                candidate[row, reject_at + 1] = (candidate[row, reject_at + 1] + 1) % logits.shape[-1]
                _, candidate_logits = replay.forward(candidate)
                batch = dataclasses.replace(state.input_batch, input_ids=candidate.flatten())
                sampled = runner.rejection_sampler(candidate_logits.flatten(0, 1).clone(), batch, None)
                count = int(sampled.num_sampled[row])
                rejected = int(sampled.num_rejected[row])
                emitted = sampled.sampled_token_ids[row, :count]
                assert count == reject_at + 1 and rejected == 6 - reject_at
                assert torch.equal(emitted, predicted[row, :count])
                valid = replay.valid_prefix(row, count)
                kv_equal = all(torch.equal(x, y[:count]) for x, y in zip(valid, prefix))
                assert kv_equal, "accepted KV differs from clean target replay"
                runner.postprocess_sampled(batch.idx_mapping, sampled.sampled_token_ids,
                                           sampled.num_sampled, sampled.num_rejected,
                                           batch.query_start_loc)
                expected_positions = batch.seq_lens - sampled.num_rejected
                computed = req.num_computed_tokens.gpu[batch.idx_mapping.long()]
                assert torch.equal(computed, expected_positions)
                from highlight_spec.speculator import prepare_first_step
                from vllm.v1.worker.gpu.input_batch import InputBuffers
                buffers = InputBuffers(max_num_reqs=6, max_num_tokens=42, device=runner.device)
                prepare_first_step(buffers, batch.seq_lens, batch.idx_mapping,
                                   sampled.num_sampled, sampled.num_rejected,
                                   req.last_sampled_tokens, req.next_prefill_tokens,
                                   runner.max_model_len, runner.max_num_reqs)
                assert torch.equal(buffers.positions[:6], expected_positions.long())
                correct_ids = req.last_sampled_tokens[batch.idx_mapping.long(), :1].int()
                assert torch.equal(buffers.input_ids[:6], correct_ids[:, 0])
                before_resume = replay.snapshot()
                _, continued = replay.forward(correct_ids, expected_positions.long())
                replay.restore(before_resume)
                replay.poison_rejected(row, count)
                _, poisoned = replay.forward(correct_ids, expected_positions.long())
                ignored = torch.equal(continued, poisoned)
                assert ignored, "rejected KV affected continuation"
                report["rejections"].append({"reject_at": reject_at, "accepted": count - 1,
                                             "rejected": rejected, "accepted_kv_equal": kv_equal,
                                             "positions_correct": True, "rejected_kv_ignored": ignored})
            report["mechanics_passed"] = all(report["verifier"][k] for k in ("hidden_bitwise_equal", "logits_bitwise_equal"))
        except Exception:
            report["error"] = traceback.format_exc()
            report["mechanics_passed"] = False
        finally:
            replay.restore(replay.original)
            restore_requests()
            state.hidden_states.copy_(original_hidden)
            selection.sparse_active = active
        return report


def main():
    args = decode.parse_args()
    assert (args.mode, args.ctx, args.batch, args.gen) == ("highlight", 130560, 6, 512)
    assert args.verify_fraction == 0 and args.reference
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    os.environ["PYTHONPATH"] = str(decode.ROOT)
    stats.WORKER_EXTENSION = "benchmarks.sanity.SanityWorker"
    reference = json.loads(Path(args.reference).read_text())["tokens"][4]
    group, _ = decode.prompt_groups(args, decode.load_prompts(args))[0]
    (out / "config.json").write_text(json.dumps({"args": vars(args),
        "source_sha256": decode.source_sha256(),
        "prompt_sha256": [decode.digest(p) for p in group]}, indent=2) + "\n")
    llm = decode.build_engine(args)
    try:
        decode.generate_batch(llm, args, group, 16)
        llm.collective_rpc("sanity_arm", kwargs={"out": str(out.resolve()), "request_id": "10", "reference": reference})
        # Keep all six requests alive at the affected round.
        outputs, _, _, _ = decode.generate_batch(llm, args, group, 128)
        (out / "tokens.json").write_text(json.dumps([list(o.outputs[0].token_ids) for o in outputs]) + "\n")
        if not (out / "checks.json").exists():
            raise RuntimeError("the saved divergence round was not reached")
        print((out / "checks.json").read_text(), flush=True)
        if not json.loads((out / "checks.json").read_text()).get("mechanics_passed"):
            raise RuntimeError("sanity checks did not all pass; see checks.json")
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
