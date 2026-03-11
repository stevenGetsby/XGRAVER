import argparse
import os
from PIL import Image
from graver.pipelines.image_to_mesh import GraverImageToMeshPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_config', type=str, default='configs/pipeline.json')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--save_npz', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_preprocess', action='store_true')
    args = parser.parse_args()

    pipeline = GraverImageToMeshPipeline.from_config(args.pipeline_config).to(args.device)
    image = Image.open(args.input)

    result = pipeline.run(
        image,
        preprocess=not args.no_preprocess,
        mesh_path=args.output,
        save_npz_path=args.save_npz or None,
        seed=args.seed,
        verbose=True,
    )

    print(f"mesh_ok: {result['mesh_ok']}")
    print(f"mesh_path: {result['mesh_path']}")
    print(f"coords_vis_path: {result['coords_vis_path']}")
    print(f"mask_vis_path: {result['mask_vis_path']}")
    print(f"normal_path: {result['normal_path']}")
    if args.save_npz:
        print(f"npz_path: {args.save_npz}")


if __name__ == '__main__':
    main()