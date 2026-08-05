#!/usr/bin/env python3
"""预处理触觉 deformation 视频：拆分拼接帧为单指视频并 resize.

输入:
  - GR00T LeRobot v2 格式数据集中的拼接触觉视频
    (observation.images.tactile_deform_right/left → 240×1200×3, 5 fingers)



输出:
  - 每个手指独立的视频文件 (224×224×3)
    observation.images.tactile_finger_0, tactile_finger_1, ...
  - 更新 meta/info.json 添加新 feature

用法:
    python examples/our_robot_df/preprocess_tactile.py \
        --dataset-path /path/to/groot_dataset \
        --num-fingers 5 \
        --finger-indices 0,1 \
        --target-size 224 \
        --source-key observation.images.tactile_deform_right
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def get_video_info(video_path: str) -> dict:
    """Get video metadata via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", video_path,
        ],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    for stream in info.get("streams", []):
        if stream["codec_type"] == "video":
            return {
                "width": int(stream["width"]),
                "height": int(stream["height"]),
                "fps": stream.get("r_frame_rate", "30/1"),
                "nb_frames": int(stream.get("nb_frames", 0)),
            }
    return {}


def split_tactile_video(
    input_video: Path,
    output_dir: Path,
    num_fingers: int,
    finger_indices: list[int],
    target_size: int,
    fps: str = "30",
) -> dict[int, Path]:
    """Split a concatenated tactile video into per-finger videos.

    Uses ffmpeg's crop + scale filters for efficient processing.
    """
    info = get_video_info(str(input_video))
    if not info:
        return {}

    width = info["width"]
    height = info["height"]
    finger_w = width // num_fingers

    output_paths = {}
    for fidx in finger_indices:
        output_path = output_dir / f"finger_{fidx}.mp4"
        crop_x = fidx * finger_w

        cmd = [
            "ffmpeg", "-y", "-i", str(input_video),
            "-vf", f"crop={finger_w}:{height}:{crop_x}:0,scale={target_size}:{target_size}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", fps,
            "-an",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] ffmpeg failed for finger {fidx}: {result.stderr[-200:]}")
            continue
        output_paths[fidx] = output_path

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Split concatenated tactile videos")
    parser.add_argument("--dataset-path", required=True, help="GR00T dataset root")
    parser.add_argument("--num-fingers", type=int, default=5,
                        help="Total fingers in concatenated image")
    parser.add_argument("--finger-indices", type=str, default="0,1",
                        help="Comma-separated finger indices to extract")
    parser.add_argument("--target-size", type=int, default=224,
                        help="Output image size (square)")
    parser.add_argument("--source-keys", type=str,
                        default="observation.images.tactile_deform_right",
                        help="Comma-separated source video key(s)")
    parser.add_argument("--fps", type=str, default="30", help="Output video FPS")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    finger_indices = [int(x.strip()) for x in args.finger_indices.split(",")]
    source_keys = [k.strip() for k in args.source_keys.split(",")]

    print(f"═══════════════════════════════════════════════")
    print(f"  触觉视频预处理: 拼接 → 单指")
    print(f"═══════════════════════════════════════════════")
    print(f"  数据集: {dataset_path}")
    print(f"  源 keys: {source_keys}")
    print(f"  手指数: {args.num_fingers}, 提取: {finger_indices}")
    print(f"  输出尺寸: {args.target_size}×{args.target_size}")
    print(f"═══════════════════════════════════════════════")
    print()

    # Find all video chunk directories
    videos_dir = dataset_path / "videos"
    if not videos_dir.exists():
        print(f"[ERROR] videos/ 目录不存在: {videos_dir}")
        return

    total_processed = 0
    output_key_map = {}  # source_key → {finger_idx: output_key}

    for source_key in source_keys:
        # Determine hand side from source key for naming
        if "right" in source_key:
            side = "right"
        elif "left" in source_key:
            side = "left"
        else:
            side = ""

        # Find source video directories (organized by chunk)
        source_dirs = sorted(videos_dir.glob(f"*/{source_key}")) + \
                      sorted(videos_dir.glob(f"{source_key}/*"))

        # Also try the flat structure: videos/source_key/chunk-xxx/
        if not source_dirs:
            source_base = videos_dir / source_key
            if source_base.exists():
                source_dirs = [source_base]

        if not source_dirs:
            print(f"[WARN] 未找到源视频目录: {source_key}")
            continue

        for source_dir in source_dirs:
            # Detect directory structure
            video_files = sorted(source_dir.glob("*.mp4"))
            chunk_dirs = sorted(source_dir.glob("chunk-*"))

            if chunk_dirs:
                # Structure: videos/source_key/chunk-xxx/file-xxx.mp4
                for chunk_dir in chunk_dirs:
                    video_files_in_chunk = sorted(chunk_dir.glob("*.mp4"))
                    for vf in video_files_in_chunk:
                        rel_chunk = chunk_dir.name  # e.g., "chunk-000"
                        for fidx in finger_indices:
                            output_key = f"observation.images.tactile_finger_{side}_{fidx}" if side else f"observation.images.tactile_finger_{fidx}"
                            out_dir = videos_dir / output_key / rel_chunk
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_path = out_dir / vf.name

                            if out_path.exists():
                                continue

                            # Split this video
                            paths = split_tactile_video(
                                vf, out_dir, args.num_fingers,
                                [fidx], args.target_size, args.fps,
                            )
                            if fidx in paths:
                                paths[fidx].rename(out_path)
                                total_processed += 1

                            if output_key not in output_key_map:
                                output_key_map[output_key] = True
            elif video_files:
                # Structure: videos/chunk-xxx/source_key/episode_xxx.mp4
                parent = source_dir.parent
                for vf in video_files:
                    for fidx in finger_indices:
                        output_key = f"observation.images.tactile_finger_{side}_{fidx}" if side else f"observation.images.tactile_finger_{fidx}"
                        out_dir = parent / output_key
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / vf.name

                        if out_path.exists():
                            continue

                        paths = split_tactile_video(
                            vf, out_dir, args.num_fingers,
                            [fidx], args.target_size, args.fps,
                        )
                        if fidx in paths:
                            paths[fidx].rename(out_path)
                            total_processed += 1

                        if output_key not in output_key_map:
                            output_key_map[output_key] = True

    print(f"\n[INFO] 处理完成: {total_processed} 个视频文件")
    print(f"[INFO] 生成 keys: {list(output_key_map.keys())}")

    # Update meta/info.json
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

        features = info.get("features", {})
        updated = False
        for output_key in output_key_map:
            if output_key not in features:
                features[output_key] = {
                    "dtype": "video",
                    "shape": [args.target_size, args.target_size, 3],
                    "names": ["height", "width", "channels"],
                    "video_info": {
                        "video.fps": float(args.fps),
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "has_audio": False,
                    },
                }
                updated = True

        if updated:
            info["features"] = features
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4)
            print(f"[INFO] meta/info.json 已更新")

    print()
    print("下一步: 更新 modality.json 使用新的 video keys:")
    for output_key in output_key_map:
        short_key = output_key.replace("observation.images.", "")
        print(f'  "{short_key}": {{"original_key": "{output_key}"}}')


if __name__ == "__main__":
    main()
