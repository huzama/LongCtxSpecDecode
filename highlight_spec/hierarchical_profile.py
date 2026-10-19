"""Opt-in CUDA-event timing of the existing captured serving path.

Install after warm-up. No graph recapture, model changes or per-stage device
synchronization. Keep these instrumented timings separate from throughput.
"""
import statistics

import torch

from .hierarchical_backend import HierarchicalStatsWorker


class StageProfile:
    def __init__(self, runner, batch, capacity):
        self.runner, self.batch = runner, batch
        self.records, self.restore = [], []
        self.current = None
        names = ("round_start", "round_end", "target_start", "target_end",
                 "sample_start", "sample_end", "draft_start", "draft_end")
        names += tuple(f"step_{i}_{edge}" for i in range(6) for edge in ("start", "end"))
        self.events = [{name:torch.cuda.Event(enable_timing=True) for name in names}
                       for _ in range(capacity)]
        # CUDA events are lazy: create their device handles outside measurement.
        for row in self.events:
            for event in row.values(): event.record()
        torch.cuda.synchronize()
        self.stream = None
        self.install()

    def mark(self, name):
        stream = torch.cuda.current_stream().cuda_stream
        if self.stream is None: self.stream = stream
        if stream != self.stream:
            raise RuntimeError("stage profiling requires one compute stream")
        self.current["events"][name].record()

    def replace(self, obj, name, factory):
        old = getattr(obj, name)
        self.restore.append((obj, name, old))
        setattr(obj, name, factory(old))

    def install(self):
        def execute(old):
            def call(output, *args, **kwargs):
                counts = list(output.num_scheduled_tokens.values())
                if len(counts) == self.batch and all(n == 7 for n in counts):
                    if self.current is not None or len(self.records) >= len(self.events):
                        raise RuntimeError("overlapping or excessive profiled rounds")
                    self.current = dict(events=self.events[len(self.records)], steps=[],
                                        target_calls=0, sampling_calls=0)
                    self.mark("round_start")
                return old(output, *args, **kwargs)
            return call

        def target(old):
            def call(desc, *args, **kwargs):
                if self.current is None: return old(desc, *args, **kwargs)
                if desc.num_tokens != self.batch*7:
                    raise RuntimeError("profiled verification graph has wrong shape")
                self.current["target_calls"] += 1
                self.mark("target_start")
                result = old(desc, *args, **kwargs)
                self.mark("target_end")
                return result
            return call

        def sample(old):
            def call(*args, **kwargs):
                if self.current is None: return old(*args, **kwargs)
                self.current["sampling_calls"] += 1
                self.mark("sample_start")
                result = old(*args, **kwargs)
                self.mark("sample_end")
                return result
            return call

        def propose(old):
            def call(*args, **kwargs):
                if self.current is None: return old(*args, **kwargs)
                self.mark("draft_start")
                result = old(*args, **kwargs)
                self.mark("draft_end")
                expected = [0, 1, 1, 1, 1, 1] if len(runner.speculator.graphs) == 2 else [0]*6
                if self.current["steps"] != expected:
                    raise RuntimeError("draft graph phases differ from the configured selector")
                return result
            return call

        def graph(phase):
            def wrap(old):
                def call(desc, *args, **kwargs):
                    if self.current is None: return old(desc, *args, **kwargs)
                    step = len(self.current["steps"])
                    if step >= 6 or desc.num_tokens != self.batch:
                        raise RuntimeError("profiled draft graph has wrong shape")
                    self.mark(f"step_{step}_start")
                    result = old(desc, *args, **kwargs)
                    self.mark(f"step_{step}_end")
                    self.current["steps"].append(phase)
                    return result
                return call
            return wrap

        def finish_round(old):
            def call(*args, **kwargs):
                result = old(*args, **kwargs)
                if self.current is not None:
                    self.mark("round_end")
                    if (self.current["target_calls"], self.current["sampling_calls"]) != (1, 1):
                        raise RuntimeError("missing verification or sampling event")
                    self.records.append(self.current)
                    self.current = None
                return result
            return call

        runner = self.runner
        self.replace(runner, "execute_model", execute)
        self.replace(runner.cudagraph_manager, "run_fullgraph", target)
        self.replace(runner, "sample", sample)
        self.replace(runner.speculator, "propose", propose)
        for phase, manager in enumerate(runner.speculator.graphs):
            self.replace(manager, "run_fullgraph", graph(phase))
        self.replace(runner, "sample_tokens", finish_round)

    def finish(self):
        torch.cuda.synchronize()
        for obj, name, old in reversed(self.restore): setattr(obj, name, old)
        if self.current is not None or not self.records:
            raise RuntimeError("incomplete stage profile")
        rows = []
        for i, record in enumerate(self.records):
            e = record["events"]
            elapsed = lambda a, b: e[a].elapsed_time(e[b])
            steps = [elapsed(f"step_{s}_start", f"step_{s}_end") for s in range(6)]
            row = dict(round=i, verification_ms=elapsed("round_start", "draft_start"),
                verification_forward_ms=elapsed("target_start", "target_end"),
                verification_sampling_ms=elapsed("sample_start", "sample_end"),
                draft_ms=elapsed("draft_start", "draft_end"), draft_graph_steps_ms=steps,
                draft_graphs_ms=sum(steps), first_draft_graph_ms=steps[0],
                reused_draft_graph_mean_ms=statistics.mean(steps[1:]),
                post_draft_ms=elapsed("draft_end", "round_end"),
                total_ms=elapsed("round_start", "round_end"))
            row["draft_preparation_ms"] = row["draft_ms"]-row["draft_graphs_ms"]
            row["verification_preparation_ms"] = (row["verification_ms"]-
                row["verification_forward_ms"]-row["verification_sampling_ms"])
            rows.append(row)
        keys = [k for k in rows[0] if k.endswith("_ms") and k != "draft_graph_steps_ms"]
        return dict(status="complete", batch=self.batch, rounds=len(rows), rows=rows,
            mean_ms={k:statistics.mean(r[k] for r in rows) for k in keys},
            median_ms={k:statistics.median(r[k] for r in rows) for k in keys},
            scope=dict(verification="target preparation, seven-query graph, logits/rejection and result preparation before drafting",
                draft="six existing draft graphs plus input/metadata preparation",
                draft_graph="full model, indexer, logits, greedy sampling and draft-input advance",
                timing="CUDA events on compute stream; includes launch gaps, excludes external scheduling and separate output-copy stream",
                synchronization="once before installation and once after generation; none between measured stages"),
            serving_throughput=False)


class HierarchicalProfileWorker(HierarchicalStatsWorker):
    def hierarchical_profile_start(self, batch=25, capacity=512):
        if getattr(self, "stage_profile", None) is not None:
            raise RuntimeError("stage profile already installed")
        self.stage_profile = StageProfile(self.model_runner, batch, capacity)
        return dict(status="installed", batch=batch)

    def hierarchical_profile_finish(self):
        result = self.stage_profile.finish()
        self.stage_profile = None
        return result
