"""Eager, batch-one quality pilot with one persistent hierarchical backbone KV.

Native HF Qwen3 weights/FFNs and the existing Highlight selector are reused.
Prefill is dense FA4. Decode reads packed planes through our Triton kernel.
This is a quality prototype, not a vLLM serving or throughput implementation.
"""
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers.cache_utils import Cache, DynamicCache

from .ffn_probe import Selector
from .ffn_quant import TensorStore
from .hierarchical_cache import HierarchicalKV
from .hierarchical_kernels import attention


class ModelCache(Cache):
    def __init__(self, config, capacity):
        super().__init__(layers=[])
        self.packed = [HierarchicalKV(capacity, config.num_key_value_heads, config.head_dim,
                                     device="cuda", encoder="triton")
                       for _ in range(config.num_hidden_layers)]

    def get_seq_length(self, layer_idx=0):
        return self.packed[layer_idx].length

    def get_mask_sizes(self, query_length, layer_idx):
        return self.get_seq_length(layer_idx)+query_length, 0

    def update(self, key, value, layer_idx, *args, **kwargs):
        cache = self.packed[layer_idx]
        if cache.length:
            cache.append(key[0].transpose(0, 1).contiguous(), value[0].transpose(0, 1).contiguous())
        # Initial prefill is encoded only after dense attention consumes raw KV.
        return key, value

    def crop(self, length):
        for cache in self.packed: cache.crop(length)

    def commit(self, length=None):
        for cache in self.packed: cache.commit(length)

    def memory_bytes(self):
        return {name:sum(cache.memory_bytes()[name] for cache in self.packed)
                for name in self.packed[0].memory_bytes()}


class Probe:
    def __init__(self, checkpoint, prompt_length, output_cap):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        raw = json.loads((Path(checkpoint)/"config.json").read_text())
        theta = raw.get("rope_theta") or raw["rope_parameters"]["rope_theta"]
        raw.update(use_dsa=False, max_position_embeddings=65536,
                   rope_parameters=dict(rope_type="yarn", factor=2.,
                       original_max_position_embeddings=32768, rope_theta=float(theta)))
        self.model = Qwen3ForCausalLM.from_pretrained(checkpoint, config=Qwen3Config(**raw),
            dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True).eval()
        ALL_ATTENTION_FUNCTIONS.register("hierarchical_probe", self.attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register("hierarchical_probe", lambda **kwargs: None)
        self.model.config._attn_implementation = "hierarchical_probe"
        self.prompt_length, self.output_cap = prompt_length, output_cap
        store = TensorStore(checkpoint)
        self.selectors = [Selector(self.model.config, store, i, prompt_length)
                          for i in range(self.model.config.num_hidden_layers)]
        self.hidden = {}
        self.hooks = []
        for i, layer in enumerate(self.model.model.layers):
            def capture(module, args, kwargs, idx=i):
                self.hidden[idx] = kwargs.get("hidden_states", args[0] if args else None)
            self.hooks.append(layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True))
        self.sparse, self.bits, self.depth, self.prefilling = False, 8, 0, False
        self.reference = False

    def attention(self, module, query, key, value, attention_mask, scaling=None, **kwargs):
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
        tokens = self.selectors[module.layer_idx].update(
            self.hidden.pop(module.layer_idx), self.positions, self.sparse, self.depth)
        q, k, v = (x.transpose(1, 2).contiguous() for x in (query, key, value))
        if self.prefilling or not isinstance(self.cache, ModelCache):
            output = _flash_attn_fwd(q, k, v, causal=True, softmax_scale=scaling, num_splits=1)[0]
            if isinstance(self.cache, ModelCache):
                self.cache.packed[module.layer_idx].prefill(k[0], v[0])
            return output, None
        cache = self.cache.packed[module.layer_idx]
        if self.reference:
            from .hierarchical_cache import reference_attention
            outputs = []
            for row in range(q.shape[1]):
                position = cache.length-q.shape[1]+row
                visible = torch.arange(position+1, device=q.device)
                key_ref, value_ref = cache.read(self.bits, visible,
                    precision_prefix=cache.precision_prefix(position))
                outputs.append(reference_attention(q[0, row:row+1], key_ref, value_ref,
                    self.positions[row:row+1], visible, scale=scaling))
            return torch.cat(outputs)[None], None
        if tokens is None: tokens = torch.arange(cache.length, device=q.device)
        return attention(cache, q[0], self.positions, tokens, self.bits, scale=scaling)[None], None

    @torch.inference_mode()
    def forward(self, ids, cache, *, sparse=False, bits=8, depth=0, prefill=False):
        self.sparse, self.bits, self.depth, self.prefilling, self.cache = sparse, bits, depth, prefill, cache
        start = cache.get_seq_length()
        self.positions = torch.arange(start, start+len(ids), device="cuda")
        result = self.model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache,
                            use_cache=True, logits_to_keep=1 if prefill else 0)
        logits = result.logits[0].float()
        if not torch.isfinite(logits).all(): raise AssertionError("nonfinite model logits")
        return logits

    def new_cache(self, packed):
        if packed: return ModelCache(self.model.config, self.prompt_length+self.output_cap+8)
        return DynamicCache(config=self.model.config)

    @torch.inference_mode()
    def generate(self, ids, *, packed, draft_bits=None, teacher=None):
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        cache = self.new_cache(packed)
        first = int(self.forward(ids, cache, prefill=True)[-1].argmax())
        output, steps, logit_agreement = [first], [], []
        while (len(output) < self.output_cap and
               (teacher[len(output)-1] if teacher is not None else output[-1]) not in eos):
            base, anchor = cache.get_seq_length(), output[-1]
            if teacher is not None:
                # Evaluate logits conditional on the same BF16 teacher prefix.
                index = len(output)-1
                anchor = teacher[index]
            if draft_bits is None:
                token = int(self.forward([anchor], cache)[-1].argmax())
                output.append(token)
                if teacher is not None:
                    logit_agreement.append(token == teacher[index+1])
                if packed: cache.commit()
                continue
            draft_count = 6
            guesses, current = [], anchor
            for depth in range(draft_count):
                token = int(self.forward([current], cache, sparse=True, bits=draft_bits, depth=depth)[-1].argmax())
                guesses.append(token)
                current = token
            cache.crop(base)
            target = self.forward([anchor]+guesses, cache, sparse=False, bits=8).argmax(-1).tolist()
            accepted = next((i for i in range(draft_count) if guesses[i] != target[i]), draft_count)
            generated = guesses[:accepted]+[target[accepted]]
            room = self.output_cap-len(output)
            stop = next((i+1 for i, token in enumerate(generated) if token in eos), len(generated))
            visible = generated[:min(room, stop)]
            uncensored = room >= draft_count+1 and not any(token in eos for token in generated)
            steps.append(dict(accepted=accepted, proposed=draft_count, emitted=len(visible),
                              uncensored=uncensored, guesses=guesses, target=target))
            output.extend(visible)
            # Anchor and accepted draft inputs are the committed target-computed KV.
            cache.commit(base+len(visible))
        record = dict(tokens=output, eos=output[-1] in eos, capped=len(output)==self.output_cap,
                      post_eos_tokens=0, rounds=steps, first_token=first)
        if steps:
            complete = [s for s in steps if s["uncensored"]]
            proposed = sum(s["proposed"] for s in complete)
            record.update(acceptance=(sum(s["accepted"] for s in complete)/proposed if proposed else None),
                          useful_tokens_per_verification=sum(s["emitted"] for s in steps)/len(steps),
                          uncensored_rounds=len(complete))
        if teacher is not None:
            record["teacher_next_token_agreement"] = sum(logit_agreement)/len(logit_agreement) if logit_agreement else None
            record["teacher_forced"] = True
            record["teacher_states"] = len(logit_agreement)
            # Conditional predictions are not an EOS-stopped generated sequence.
            for key in ("tokens", "eos", "capped", "post_eos_tokens", "rounds", "first_token"):
                record.pop(key, None)
        if packed: record["backbone_cache_bytes"] = cache.memory_bytes()
        self.cache = None
        del cache
        gc.collect()
        torch.cuda.empty_cache()
        return record


def pilot(args):
    from benchmarks.hierarchical_cache import save
    from transformers import AutoTokenizer
    torch.set_num_threads(8)
    torch.manual_seed(42)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    # The campaign manifest freezes token IDs, not a mutable dataset query.
    prompts = manifest if isinstance(manifest, list) else manifest["prompts"]
    samples = prompts[:args.samples]
    if len(samples) != args.samples or args.samples < 1 or not 8 <= args.tokens <= 128:
        raise ValueError("pilot requires frozen prompts and an 8..128 token cap")
    records = []
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    probe = None
    plan = dict(stage="quality prototype", manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                checkpoint=args.checkpoint, samples=args.samples, output_cap=args.tokens,
                config_sha256=hashlib.sha256((Path(args.checkpoint)/"config.json").read_bytes()).hexdigest(),
                runtime={name:importlib.metadata.version(name) for name in ("torch", "triton", "transformers", "vllm")},
                prompt_hashes=[p["prompt_sha256"] for p in samples],
                prefill="one full dense BF16 FA4 prefill; compress each layer after its attention",
                draft_tokens=6, draft_fraction=.07, recent_selection_tokens=64,
                boundary_policy="six drafts; query-position-specific compression view inside verification",
                quantization_group=128, recent_precision="BF16", ffn="BF16", indexer="BF16",
                cache="one hierarchical backbone cache plus existing indexer cache",
                variants=["bf16-ar", "hierarchical8-ar", "highlight8-verify8", "highlight4-verify8"],
                backend="HF Qwen3 + FA4 prefill + Triton packed attention", serving_throughput=False)
    save(args.out/"plan.json", plan)
    for slot, item in enumerate(samples):
        ids = item["prompt_token_ids"]
        if hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest() != item["prompt_sha256"]:
            raise ValueError("frozen prompt token hash differs")
        if len(ids) != 32768:
            raise ValueError("first pilot is fixed at the existing 32K context")
        if probe is None: probe = Probe(args.checkpoint, len(ids), args.tokens)
        row = dict(slot=slot, variants={})
        for name, packed, bits in (("bf16-ar", False, None), ("hierarchical8-ar", True, None),
                                   ("highlight8-verify8", True, 8), ("highlight4-verify8", True, 4)):
            result = probe.generate(ids, packed=packed, draft_bits=bits)
            result["text"] = tokenizer.decode(result["tokens"], skip_special_tokens=False)
            row["variants"][name] = result
            print(json.dumps(dict(slot=slot, variant=name, acceptance=result.get("acceptance"),
                                  tokens=len(result["tokens"]))), flush=True)
            save(args.out/f"sample-{slot}.json", row)
        teacher = row["variants"]["bf16-ar"]["tokens"]
        # Only score states that existed before the BF16 teacher's EOS/cap.
        if len(teacher) == args.tokens:
            check = probe.generate(ids, packed=True, teacher=teacher)
            row["target_teacher_agreement"] = check["teacher_next_token_agreement"]
        target = row["variants"]["hierarchical8-ar"]["tokens"]
        row["speculative_matches_8bit_ar"] = {
            name: value["tokens"] == target for name, value in row["variants"].items() if "verify8" in name}
        save(args.out/f"sample-{slot}.json", row)
        records.append(row)
    save(args.out/"pilot.json", dict(status="complete", samples=records, serving_throughput=False,
        quality_scope="short capped continuations and teacher agreement; no task-accuracy claim"))


@torch.inference_mode()
def diagnose(args):
    """Compare serial and block verification on exactly the same token prefix.

    Use the first divergent round in saved evidence. Neither path commits during
    the compared forward passes, so cache compression cannot change between them.
    A BF16 FA4 control distinguishes ordinary batch-shape numerics from packing.
    """
    from benchmarks.hierarchical_cache import save
    if args.pilot is None:
        raise ValueError("diagnose requires a completed --pilot directory")
    records = json.loads((args.pilot/"pilot.json").read_text())["samples"]
    manifest = json.loads(Path(args.manifest).read_text())
    prompts = manifest if isinstance(manifest, list) else manifest["prompts"]
    case = None
    for record in records:
        target = record["variants"]["hierarchical8-ar"]["tokens"]
        spec = record["variants"]["highlight8-verify8"]
        mismatch = next((i for i, (a, b) in enumerate(zip(target, spec["tokens"])) if a != b), None)
        if mismatch is not None:
            start = 1
            for step in spec["rounds"]:
                if start <= mismatch < start + step["emitted"]:
                    case = (record["slot"], target, start, mismatch, step, spec)
                    break
                start += step["emitted"]
        if case is not None: break
    if case is None:
        save(args.out/"diagnosis.json", dict(status="no saved mismatch"))
        return
    slot, target, start, mismatch, step, spec = case
    ids = prompts[slot]["prompt_token_ids"]
    probe = Probe(args.checkpoint, len(ids), args.tokens)
    inputs = [target[start-1]] + step["guesses"]
    result = dict(slot=slot, first_output_mismatch=mismatch, round_output_start=start,
                  inputs=inputs, controls={}, serving_throughput=False)
    for name, packed in (("hierarchical8_serial_prefix", True),
                         ("hierarchical8_verified_prefix", True), ("bf16_fa4", False),
                         ("bf16_verified_prefix", False), ("hierarchical8_reference", True)):
        cache = probe.new_cache(packed)
        probe.forward(ids, cache, prefill=True)
        if name == "hierarchical8_reference":
            # Diagnostic only: fixed-reduction native GEMM plus independent
            # single-query FP32 attention. The serving/pilot path is unchanged.
            from types import MethodType
            from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
            def invariant_forward(module, x):
                shape = x.shape
                out = linear_batch_invariant(x.reshape(-1, shape[-1]), module.weight, module.bias)
                return out.reshape(*shape[:-1], module.out_features)
            for module in probe.model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.forward = MethodType(invariant_forward, module)
            probe.reference = True
        replay_matches = []
        if name.endswith("verified_prefix"):
            offset = 1
            for prior in spec["rounds"]:
                if offset >= start: break
                base = cache.get_seq_length()
                logits = probe.forward([spec["tokens"][offset-1]]+prior["guesses"], cache)
                replay_matches.append(logits.argmax(-1).tolist() == prior["target"])
                if packed: cache.commit(base+prior["emitted"])
                else: cache.crop(base+prior["emitted"])
                offset += prior["emitted"]
        else:
            for token in target[:start-1]:
                probe.forward([token], cache)
                if packed: cache.commit()
        base = cache.get_seq_length()
        block = probe.forward(inputs, cache)
        block_tail = [torch.stack([t[base-c.prefix:base-c.prefix+len(inputs)].clone() for t in c.tail])
                      for c in cache.packed] if packed else []
        cache.crop(base)
        serial = torch.cat([probe.forward([token], cache) for token in inputs])
        differences = []
        if packed:
            for c, old in zip(cache.packed, block_tail):
                now = torch.stack([t[base-c.prefix:base-c.prefix+len(inputs)] for t in c.tail])
                differences.append(float((now.float()-old.float()).abs().max()))
        def top(logits):
            values, tokens = logits.topk(5)
            return dict(tokens=tokens.tolist(), logits=values.tolist(),
                        top_two_margin=(values[:, 0]-values[:, 1]).tolist())
        result["controls"][name] = dict(base_length=base,
            replayed_rounds_match_saved=replay_matches,
            prefix=cache.packed[0].prefix if packed else None,
            block=top(block), serial=top(serial),
            differing_positions=(block.argmax(-1) != serial.argmax(-1)).nonzero().flatten().tolist(),
            logits_bitwise_equal=torch.equal(block, serial),
            max_abs_logit_difference=(block-serial).abs().amax(-1).tolist(),
            per_layer_max_abs_tail_difference=differences)
        save(args.out/"diagnosis.json", result)
        print(json.dumps(dict(control=name, differing_positions=result["controls"][name]["differing_positions"])), flush=True)
        probe.cache = None
        del cache, block_tail
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "complete"
    save(args.out/"diagnosis.json", result)
