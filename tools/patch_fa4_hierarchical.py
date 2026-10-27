#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create an isolated FA4 overlay with a packed-KV loader factory hook.

The installed package is never changed. Prepend the output directory to
PYTHONPATH when testing the hierarchical backend. The original package source,
its license, the small applied patch and source hashes accompany the overlay.
"""

import argparse
import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
from pathlib import Path

SOURCE_SHA256 = "24ff53a4d10d92ac0f86f3484cc978462686b795b62fd15c13d7c3cb001889b3"
SOURCE_FILE = Path("cute/flash_fwd_sm100.py")
MARKER = "hierarchical_loader_hook.json"


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"FA4 hook anchor is not unique: {before[:80]!r}")
    return source.replace(before, after, 1)


def patched_source(source):
    source = replace_once(
        source,
        "                tile_scheduler=tile_scheduler,\n            )\n\n"
        "        # ///////////////////////////////////////////////////////////////////////////////\n"  # noqa: E501
        "        #  MMA",
        "                tile_scheduler=tile_scheduler,\n"
        "                aux_data=self.prepare_load_aux_data(aux_data, storage),\n"
        "            )\n\n"
        "        # ///////////////////////////////////////////////////////////////////////////////\n"  # noqa: E501
        "        #  MMA",
    )
    source = replace_once(
        source,
        "        tile_scheduler: TileSchedulerProtocol,\n    ):\n"
        "        num_load_threads = len(self.load_warp_ids)",
        "        tile_scheduler: TileSchedulerProtocol,\n"
        "        aux_data: AuxData = AuxData(),\n    ):\n"
        "        num_load_threads = len(self.load_warp_ids)",
    )
    source = replace_once(
        source,
        "                paged_kv_manager = PagedKVManager.create(\n",
        "                paged_kv_manager = self.make_paged_kv_manager(\n",
    )
    source = replace_once(
        source,
        "                    mK.element_type,\n                )\n"
        "                tKsK, tKgK = None, None",
        "                    mK.element_type,\n"
        "                    aux_data=aux_data,\n                )\n"
        "                tKsK, tKgK = None, None",
    )
    factory = '''    def extend_shared_storage(self, storage):
        return storage

    def prepare_load_aux_data(self, aux_data, storage):
        return aux_data

    def make_paged_kv_manager(
        self, mPageTable, mK, mV, page_size_divmod, batch, head, thread,
        seqlen, leftpad, tile_n, head_dim, head_dim_v, threads, dtype,
        *, aux_data=AuxData(),
    ):
        """Default paged loader; subclasses may consume the existing aux data."""
        return PagedKVManager.create(
            mPageTable, mK, mV, page_size_divmod, batch, head, thread,
            seqlen, leftpad, tile_n, head_dim, head_dim_v, threads, dtype,
        )

'''
    source = replace_once(
        source,
        "        self.shared_storage = SharedStorage\n",
        "        self.shared_storage = self.extend_shared_storage(SharedStorage)\n",
    )
    return replace_once(
        source,
        "    @cute.jit\n    def load(\n",
        factory + "    @cute.jit\n    def load(\n",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="installed flash_attn package directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new overlay directory"
    )
    args = parser.parse_args()
    if args.source is None:
        spec = importlib.util.find_spec("flash_attn")
        if spec is None or not spec.submodule_search_locations:
            parser.error("flash_attn is not installed; provide --source")
        args.source = Path(next(iter(spec.submodule_search_locations)))
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        parser.error("output must be outside the installed package")
    original_bytes = (source / SOURCE_FILE).read_bytes()
    actual = hashlib.sha256(original_bytes).hexdigest()
    if actual != SOURCE_SHA256:
        parser.error(f"unreviewed FA4 source {actual}; expected {SOURCE_SHA256}")
    original = original_bytes.decode()
    patched = patched_source(original)
    patched_hash = hashlib.sha256(patched.encode()).hexdigest()
    manifest = dict(
        source_sha256=actual,
        patched_sha256=patched_hash,
        package="flash-attn-4==4.0.0b33",
        source=str(source),
        hook_version=2,
    )
    if output.exists():
        marker = output / MARKER
        target = output / "flash_attn" / SOURCE_FILE
        if (
            marker.exists()
            and target.exists()
            and json.loads(marker.read_text()) == manifest
            and hashlib.sha256(target.read_bytes()).hexdigest() == patched_hash
        ):
            print(output)
            return
        parser.error("output already exists and is not this exact overlay")
    output.mkdir(parents=True)
    shutil.copytree(
        source,
        output / "flash_attn",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (output / "flash_attn" / SOURCE_FILE).write_text(patched)
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile="a/flash_attn/" + str(SOURCE_FILE),
        tofile="b/flash_attn/" + str(SOURCE_FILE),
    )
    (output / "hierarchical-loader.patch").write_text("".join(diff))
    (output / MARKER).write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        dist = importlib.metadata.distribution("flash-attn-4")
    except importlib.metadata.PackageNotFoundError:
        dist = None
    if dist is not None:
        for entry in dist.files or ():
            if entry.name.upper().startswith(("LICENSE", "COPYING")):
                dest = output / "licenses" / str(entry)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dist.locate_file(entry), dest)
    print(output)


if __name__ == "__main__":
    main()
