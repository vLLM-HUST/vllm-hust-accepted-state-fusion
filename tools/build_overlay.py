# SPDX-License-Identifier: Apache-2.0
"""Build the measured experimental overlay from the exact admitted R10 SDK."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = b'''@triton.jit(do_not_specialize=["N"])
def _publish_accepted(OUT, SEATS, SELECTION, N, CAP: tl.constexpr, ROWS: tl.constexpr):
    row = tl.arange(0, ROWS)
    live = row < N
    seat = tl.load(SEATS + row, live, other=0)
    accepted = tl.load(SELECTION + seat, live, other=1)
    tl.store(OUT + row, accepted, row < CAP)


_qualified_publication_logged = False

def publish_accepted(output, seats, selection, num_reqs, *, record=True):
    global _qualified_publication_logged
    capacity = output.numel()
    _publish_accepted[(1,)](
        output, seats, selection, num_reqs,
        CAP=capacity, ROWS=triton.next_power_of_2(capacity), num_warps=4,
    )
    if record and not _qualified_publication_logged:
        print("R10_RESEARCH_ACCEPTED_STATE_PUBLICATION", flush=True)
        _qualified_publication_logged = True


'''
USE_KERNEL = b"""    publish_accepted(
        runner.num_accepted_tokens.gpu, state.seats, root.conv_selection.tensor, n
    )
"""
COMPILE_BEFORE_SEAL = b"""    if hasattr(runner, '_live_state_root'):
        from .state_address import _publish_accepted, publish_accepted

        with compile_only(_publish_accepted):
            publish_accepted(
                runner.num_accepted_tokens.gpu, runner._live_ingress.seats,
                runner._live_state_root.conv_selection.tensor, 1, record=False,
            )
        variants['accepted_state_publication'] = 1
"""


def verify(raw, expected, label):
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {label}")


def edit_runtime(raw):
    # Edit by line position only after the complete input hash has matched.
    lines = raw.splitlines(keepends=True)
    lines[99:103] = [USE_KERNEL]
    lines.insert(86, KERNEL)
    lines.insert(7, b"from vllm.triton_utils import triton, tl\n")
    return b"".join(lines)


def edit_warmup(raw):
    lines = raw.splitlines(keepends=True)
    lines.insert(133, COMPILE_BEFORE_SEAL)
    return b"".join(lines)


def edit_contract(raw, files):
    for name in ("state_address", "serving_warmup"):
        entry = files[name]
        before = entry["source_sha256"].encode()
        if raw.count(before) != 1:
            raise ValueError(f"Expected one host-contract pin for {name}")
        raw = raw.replace(before, entry["overlay_sha256"].encode(), 1)
    return raw


def build(sdk: Path, output: Path):
    metadata = json.loads((ROOT / "mod.json").read_bytes())
    sdk, output = sdk.resolve(strict=True), output.resolve()
    if output.is_relative_to(sdk):
        raise ValueError("Choose a fresh output directory outside the SDK")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    files = metadata["files"]
    originals = {}
    for name, entry in files.items():
        source = sdk / entry["path"]
        raw = source.read_bytes()
        verify(raw, entry["source_sha256"], entry["path"])
        originals[name] = raw

    overlay = {
        "state_address": edit_runtime(originals["state_address"]),
        "serving_warmup": edit_warmup(originals["serving_warmup"]),
        "host_contract": edit_contract(originals["host_contract"], files),
    }
    payloads = {}
    manifest = {"id": metadata["id"], "engine": metadata["engine"], "overlays": []}
    for name, raw in overlay.items():
        entry = files[name]
        filename = Path(entry["path"]).name
        verify(raw, entry["overlay_sha256"], filename)
        if filename.endswith(".py"):
            compile(raw, filename, "exec")
        else:
            json.loads(raw)
        payloads[filename] = raw
        manifest["overlays"].append({"path": entry["path"], "file": filename,
                                     "original_sha256": entry["source_sha256"],
                                     "candidate_sha256": entry["overlay_sha256"]})
    payloads["overlay-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.mkdir()
    written = []
    try:
        for name, raw in payloads.items():
            path = output / name
            with path.open("xb") as stream:
                written.append(path)
                stream.write(raw)
    except BaseException:
        for path in written:
            path.unlink()
        output.rmdir()
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk", type=Path, help="Root of the admitted R10 SDK")
    parser.add_argument("output", type=Path, help="Fresh directory outside the SDK")
    args = parser.parse_args()
    try:
        manifest = build(args.sdk, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    for entry in manifest["overlays"]:
        print(f"{entry['candidate_sha256']}  {args.output / entry['file']}")


if __name__ == "__main__":
    main()
