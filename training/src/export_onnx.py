"""Export an AgroStake PyTorch checkpoint to ONNX.

The exported graph uses a static batch of 1 and NCHW input, which is the
standard ONNX form accepted by ESP-PPQ.  ESP-DL converts the graph to its
NHWC runtime layout during quantisation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .models import build_model, trainable_parameters
except ImportError:  # pragma: no cover
    from models import build_model, trainable_parameters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=["v2", "gray", "binary"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--dynamic-batch", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    channels = 1 if args.arch == "gray" else 3
    num_classes = 2 if args.arch == "binary" else 4
    out = Path(args.out) if args.out else Path(f"agrostake_{args.arch}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args.arch, num_classes=num_classes).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    dummy = torch.randn(1, channels, args.img_size, args.img_size)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if args.dynamic_batch else None
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    meta = {
        "arch": args.arch,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "onnx": str(out.resolve()),
        "input_name": "input",
        "input_shape": [1, channels, args.img_size, args.img_size],
        "output_name": "logits",
        "output_shape": [1, num_classes],
        "opset": args.opset,
        "dynamic_batch": bool(args.dynamic_batch),
        "trainable_params": trainable_parameters(model),
        "class_names": (
            ["Healthy", "Diseased"]
            if args.arch == "binary"
            else ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]
        ),
        "note": "NCHW ONNX; ESP-PPQ/ESP-DL performs the NHWC runtime layout conversion.",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if not args.no_verify:
        import onnxruntime as ort

        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        reference = model(dummy).detach().numpy()
        got = session.run(None, {session.get_inputs()[0].name: dummy.numpy()})[0]
        max_abs = float(np.max(np.abs(reference - got)))
        print(f"onnxruntime parity: max_abs_diff={max_abs:.3e}")
        if max_abs > 1e-4:
            raise SystemExit(f"ONNX parity check failed: max_abs_diff={max_abs}")
        print(f"input={session.get_inputs()[0].shape} output={session.get_outputs()[0].shape}")

    print(f"wrote {out} ({out.stat().st_size:,} bytes) and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
