"""
make_videos.py
--------------
Compile the per-epoch PNG frames saved during training into MP4 animations.

Usage:
    python make_videos.py [--fps 2] [--num_hyperparams 3]

Output structure:
    outputs/<idx>/objectness_img<i>_level<l>.mp4
    outputs/<idx>/object_proposals_img<i>.mp4
    outputs/<idx>/bb_assignments_img<i>.mp4
"""

import argparse
import glob
import os
import re
import sys

from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageDraw


# def frames_to_video(frame_paths: list[str], out_path: str, fps: int) -> None:
#     """Write a list of image paths to an MP4 video."""
#     if not frame_paths:
#         print(f"  [skip] no frames found for {out_path}")
#         return

#     first = np.array(Image.open(frame_paths[0]).convert("RGB"))
#     h, w = first.shape[:2]
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

#     for path in frame_paths:
#         frame = np.array(Image.open(path).convert("RGB"))
#         # Resize if any frame differs in size (shouldn't happen, but be safe)
#         if frame.shape[:2] != (h, w):
#             frame = np.array(Image.fromarray(frame).resize((w, h)))
#         writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

#     writer.release()
#     print(f"  Saved → {out_path}  ({len(frame_paths)} frames @ {fps} fps)")


def compile_objectness(base_dir: str, fps: int) -> None:
    """
    Compile objectness heatmap PNGs into a 5x1 grid.

    Frame filenames:  epoch{e:03d}_img{i}_level{l}.png
    Output:           objectness_img{i}.gif
    """
    pattern = os.path.join(base_dir, "objectness", "epoch*_img*_level*.png")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        print("  [objectness] no files found, skipping.")
        return

    # Group files by img_idx and epoch
    data = defaultdict(lambda: defaultdict(list))
    for f in all_files:
        m = re.search(r"epoch(\d+)_img(\d+)_level(\d+)\.png", os.path.basename(f))
        if m:
            epoch = int(m.group(1))
            img_idx = int(m.group(2))
            level = int(m.group(3))
            data[img_idx][epoch].append((level, f))

    for img_idx in sorted(data.keys()):
        if img_idx >= 8: continue
        composite_frames = []
        for epoch in sorted(data[img_idx].keys()):
            files = data[img_idx][epoch]
            files.sort(key=lambda x: x[0])
            imgs = []
            for lvl, p in files:
                imgs.append({'level': lvl, 'img': Image.open(p).convert("RGB")})
            if not imgs: continue
            
            target_width = max(im['img'].size[0] for im in imgs)
            target_height = max(im['img'].size[1] for im in imgs)
            
            imgs_resized = []
            for obj in imgs:
                im = obj['img'].resize((target_width, target_height), Image.Resampling.LANCZOS)
                im_np = np.array(im)
                text = f"Level: {obj['level']}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(im_np, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
                cv2.putText(im_np, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                imgs_resized.append(Image.fromarray(im_np))
                
            total_width = sum(im.size[0] for im in imgs_resized)
            
            text = f"Epoch: {epoch} | Img: {img_idx}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            header_height = th + 40
            
            composite = Image.new("RGB", (total_width, target_height + header_height), color=(0, 0, 0))
            x_offset = 0
            for im in imgs_resized:
                composite.paste(im, (x_offset, header_height))
                x_offset += im.size[0]
                
            composite_np = np.array(composite)
            
            x_text = (total_width - tw) // 2
            y_text = (header_height + th) // 2 - 5
            cv2.putText(composite_np, text, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            composite_frames.append(Image.fromarray(composite_np))
            
        if not composite_frames: continue
        
        out = os.path.join(base_dir, f"objectness_img{img_idx}.gif")
        duration = int(1000 / fps)
        composite_frames[0].save(
            out,
            save_all=True,
            append_images=composite_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"  Saved → {out}  ({len(composite_frames)} frames @ {fps} fps)")


def compile_proposals(base_dir: str, fps: int) -> None:
    """
    Compile object proposal PNGs.

    Frame filenames:  epoch{e:03d}_img{i}.png
    Output:           object_proposals_img{i}.mp4
    """
    pattern = os.path.join(base_dir, "object_proposals", "epoch*_img*.png")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        print("  [proposals] no files found, skipping.")
        return

    img_indices = set()
    for f in all_files:
        m = re.search(r"epoch\d+_img(\d+)\.png", os.path.basename(f))
        if m:
            img_indices.add(int(m.group(1)))

    for img_idx in sorted(img_indices):
        if img_idx >= 8: continue
        frames = sorted(
            f for f in all_files
            if re.search(rf"epoch\d+_img{img_idx}\.png", os.path.basename(f))
        )
        out = os.path.join(base_dir, f"object_proposals_img{img_idx}.gif")
        frames_to_gif(frames, out, fps)


def compile_assignments(base_dir: str, fps: int) -> None:
    """
    Compile anchor assignment PNGs.

    Frame filenames:  epoch{e:03d}_img{i}.png
    Output:           bb_assignments_img{i}.mp4
    """
    pattern = os.path.join(base_dir, "bb_assignments", "epoch*_img*.png")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        print("  [assignments] no files found, skipping.")
        return

    img_indices = set()
    for f in all_files:
        m = re.search(r"epoch\d+_img(\d+)\.png", os.path.basename(f))
        if m:
            img_indices.add(int(m.group(1)))

    for img_idx in sorted(img_indices):
        if img_idx >= 8: continue
        frames = sorted(
            f for f in all_files
            if re.search(rf"epoch\d+_img{img_idx}\.png", os.path.basename(f))
        )
        out = os.path.join(base_dir, f"bb_assignments_img{img_idx}.gif")
        frames_to_gif(frames, out, fps)

def compile_roi_heads(base_dir: str, fps: int) -> None:
    """
    Compile ROI Heads PNGs.

    Frame filenames:  epoch{e:03d}_img{i}.png
    Output:           roi_head_outputs_img{i}.mp4
    """
    pattern = os.path.join(base_dir, "roi_head_outputs", "epoch*_img*.png")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        print("  [roi_heads] no files found, skipping.")
        return

    img_indices = set()
    for f in all_files:
        m = re.search(r"epoch\d+_img(\d+)\.png", os.path.basename(f))
        if m:
            img_indices.add(int(m.group(1)))

    for img_idx in sorted(img_indices):
        if img_idx >= 8: continue
        frames = sorted(
            f for f in all_files
            if re.search(rf"epoch\d+_img{img_idx}\.png", os.path.basename(f))
        )
        out = os.path.join(base_dir, f"roi_head_outputs_img{img_idx}.gif")
        frames_to_gif(frames, out, fps)


def frames_to_gif(frame_paths: list[str], out_path: str, fps: int) -> None:
    """Write a list of image paths to a GIF."""
    if not frame_paths:
        print(f"  [skip] no frames found for {out_path}")
        return

    images = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        
        m_epoch = re.search(r"epoch(\d+)", os.path.basename(path))
        m_img = re.search(r"img(\d+)", os.path.basename(path))
        if m_epoch and m_img:
            epoch = int(m_epoch.group(1))
            img_num = int(m_img.group(1))
            img_np = np.array(img).copy()
            text = f"Epoch: {epoch} | Img: {img_num}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(img_np, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
            cv2.putText(img_np, text, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            img = Image.fromarray(img_np)
            
        images.append(img)

    duration = int(1000 / fps)  # milliseconds per frame

    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    print(f"  Saved → {out_path}  ({len(frame_paths)} frames @ {fps} fps)")

def main():
    parser = argparse.ArgumentParser(description="Compile training PNGs into MP4 animations.")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    args = parser.parse_args()

    # Auto-discover hyperparam directories: any subdir of outputs/ that is NOT 'images'
    if not os.path.isdir("outputs"):
        print("'outputs/' not found. Run training first.")
        return

    hyperparam_dirs = sorted(
        d for d in os.listdir("outputs")
        if os.path.isdir(os.path.join("outputs", d)) and d != "images"
    )
    if not hyperparam_dirs:
        print("No hyperparam output directories found in 'outputs/'.")
        return

    for hname in hyperparam_dirs:
        base_dir = os.path.join("outputs", hname)
        print(f"\n=== {hname} ===")
        compile_objectness(base_dir, args.fps)
        compile_proposals(base_dir, args.fps)
        compile_assignments(base_dir, args.fps)
        compile_roi_heads(base_dir, args.fps)

    print("\nDone.")


if __name__ == "__main__":
    main()
