import shutil
import time
from tqdm import tqdm
import cv2
import h5py
import argparse
import os
import sys
import numpy as np
import openslide
import torch
from PIL import ImageDraw
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.ops import unary_union
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def segment_tissue(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mthresh = 7
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
    _, img_prepped = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_prepped = cv2.morphologyEx(img_prepped, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    contours, hierarchy = cv2.findContours(
        img_prepped, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    return contours, hierarchy


def detect_foreground(contours, hierarchy):
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    # find foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    foreground_contours = [contours[cont_idx] for cont_idx in hierarchy_1]

    all_holes = []
    for cont_idx in hierarchy_1:
        all_holes.append(np.flatnonzero(hierarchy[:, 1] == cont_idx))

    hole_contours = []
    for hole_ids in all_holes:
        holes = [contours[idx] for idx in hole_ids]
        hole_contours.append(holes)

    return foreground_contours, hole_contours


def construct_polygon(foreground_contours, hole_contours, min_area):
    polys = []
    for foreground, holes in zip(foreground_contours, hole_contours):
        # We remove all contours that consist of fewer than 3 points, as these won't work with the Polygon constructor.
        if len(foreground) < 3:
            continue

        # remove redundant dimensions from the contour and convert to Shapely Polygon
        poly = Polygon(np.squeeze(foreground))

        # discard all polygons that are considered too small
        if poly.area < min_area:
            continue

        if not poly.is_valid:
            # This is likely becausee the polygon is self-touching or self-crossing.
            # Try and 'correct' the polygon using the zero-length buffer() trick.
            # See https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
            poly = poly.buffer(0)

        # Punch the holes in the polygon
        for hole_contour in holes:
            if len(hole_contour) < 3:
                continue

            hole = Polygon(np.squeeze(hole_contour))

            if not hole.is_valid:
                continue

            # ignore all very small holes
            if hole.area < min_area:
                continue

            poly = poly.difference(hole)

        polys.append(poly)

    if len(polys) == 0:
        raise Exception("Raw tissue mask consists of 0 polygons")

    # If we have multiple polygons, we merge any overlap between them using unary_union().
    # This will result in a Polygon or MultiPolygon with most tissue masks.
    return unary_union(polys)


def generate_tiles(
    tile_width_pix, tile_height_pix, img_width, img_height, offsets=[(0, 0)]
):
    # Generate tiles covering the entire image.
    # Provide an offset (x,y) to create a stride-like overlap effect.
    # Add an additional tile size to the range stop to prevent tiles being cut off at the edges.
    range_stop_width = int(np.ceil(img_width + tile_width_pix))
    range_stop_height = int(np.ceil(img_height + tile_height_pix))

    rects = []
    for xmin, ymin in offsets:
        cols = range(int(np.floor(xmin)), range_stop_width, tile_width_pix)
        rows = range(int(np.floor(ymin)), range_stop_height, tile_height_pix)
        for x in cols:
            for y in rows:
                rect = Polygon(
                    [
                        (x, y),
                        (x + tile_width_pix, y),
                        (x + tile_width_pix, y - tile_height_pix),
                        (x, y - tile_height_pix),
                    ]
                )
                rects.append(rect)
    return rects



def make_tile_QC_fig(tiles, slide, level, line_width_pix=1, extra_tiles=None, names=None, out_dir=None):
    # Render the tiles on an image derived from the specified zoom level
    img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    downsample = 1 / slide.level_downsamples[level]

    if names is not None:
        for tile, name in tqdm(zip(tiles, names)):
            save_path = os.path.join(out_dir, name)
            bbox = tuple(np.array(tile.bounds) * downsample)
            tile_img = img.crop(bbox).convert("RGB")
            tile_img.save(save_path, "JPEG")
    return img

def create_tissue_tiles(
    wsi, tissue_mask_scaled, tile_size_microns, offsets_micron=None
):
    print(f"Desired tile size is {tile_size_microns} um")

    # Compute the tile size in pixels from the desired tile size in microns and the image resolution
    assert (
        openslide.PROPERTY_NAME_MPP_X in wsi.properties
    ), "microns per pixel along X-dimension not available"
    assert (
        openslide.PROPERTY_NAME_MPP_Y in wsi.properties
    ), "microns per pixel along Y-dimension not available"

    mpp_x = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
    mpp_y = float(wsi.properties[openslide.PROPERTY_NAME_MPP_Y])
    mpp = min(mpp_x, mpp_y)  #  使用较小的MPP

    target_level = 0
    target_tile_size_pixels = int(tile_size_microns / mpp) #  目标像素大小

    #  找到最接近目标像素大小的level
    for i in range(wsi.level_count):
        level_tile_size = int(target_tile_size_pixels / wsi.level_downsamples[i])
        if level_tile_size >= 64:  # 最小 tile size
            target_level = i
            break

    print(f"Using level {target_level} for tiling.")

    tile_size_pix = int(tile_size_microns / (mpp * wsi.level_downsamples[target_level]))  #  在该 level 的像素大小

    # Use the tissue mask bounds as base offsets (+ a margin of a few tiles) to avoid wasting CPU power creating tiles that are never going
    # to be inside the tissue mask.
    tissue_margin_pix = tile_size_pix * 2
    minx, miny, maxx, maxy = tissue_mask_scaled.bounds
    min_offset_x = minx - tissue_margin_pix
    min_offset_y = miny - tissue_margin_pix

    offsets = [(min_offset_x, min_offset_y)]

    if offsets_micron is not None:
        assert (
            len(offsets_micron) > 0
        ), "offsets_micron needs to contain at least one value"
        # Compute the offsets in micron scale
        mpp_scale_factor = mpp
        offset_pix = [round(o / mpp_scale_factor) for o in offsets_micron]
        offsets = [(o + min_offset_x, o + min_offset_y) for o in offset_pix]

    # Generate tiles covering the entire WSI
    all_tiles = generate_tiles(
        tile_size_pix,
        tile_size_pix,
        maxx + tissue_margin_pix,
        maxy + tissue_margin_pix,
        offsets=offsets,
    )

    # Retain only the tiles that sit within the tissue mask polygon
    filtered_tiles = [rect for rect in all_tiles if tissue_mask_scaled.intersects(rect)]
    names = [
        f"{int(rect.bounds[0])}_{int(rect.bounds[1])}.jpg"  # 提取左上角坐标
        for rect in all_tiles
        if tissue_mask_scaled.intersects(rect)
    ]

    return filtered_tiles, names

def create_tissue_mask(wsi, seg_level):
    # Determine the best level to determine the segmentation on
    level_dims = wsi.level_dimensions[seg_level]

    img = np.array(wsi.read_region((0, 0), seg_level, level_dims))

    # Get the total surface area of the slide level that was used
    level_area = level_dims[0] * level_dims[1]

    # Minimum surface area of tissue polygons (in pixels)
    # Note that this value should be sensible in the context of the chosen tile size
    min_area = level_area / 10000

    contours, hierarchy = segment_tissue(img)
    foreground_contours, hole_contours = detect_foreground(contours, hierarchy)
    tissue_mask = construct_polygon(foreground_contours, hole_contours, min_area)

    # Scale the tissue mask polygon to be in the coordinate space of the slide's level 0
    scale_factor = wsi.level_downsamples[seg_level]
    tissue_mask_scaled = scale(
        tissue_mask, xfact=scale_factor, yfact=scale_factor, zfact=1.0, origin=(0, 0)
    )

    return tissue_mask_scaled








if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Preprocessing script")
    parser.add_argument("--input_slide", type=str,
                        help="Path to input WSI file",)
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output data",
    )

    parser.add_argument(
        "--tile_size",
        help="Desired tile size in microns (should be the same value as used in feature extraction model).",
        type=int,
        default=360
    )
    args = parser.parse_args()

    # Derive the slide ID from its name
    slide_id, _ = os.path.splitext(os.path.basename(args.input_slide))

    os.makedirs(args.output_dir, exist_ok=True)

    path = os.path.join(args.output_dir, slide_id)
    if os.path.exists(path):
        print(f"路径 {path} 已经存在，程序退出。")
        # shutil.move(args.input_slide, args.input_slide.replace('child_path-svs', 'child_path-svs-back_up'))
        sys.exit()
    else:
        # 路径不存在，创建目录
        os.makedirs(path, exist_ok=True)
        print(f"成功创建目录 {path}")
    os.makedirs(os.path.join(args.output_dir, slide_id, 'tiles'), exist_ok=True)


    # Open the slide for reading
    wsi = openslide.open_slide(args.input_slide)
    # Get the dimensions
    width, height = wsi.dimensions
    print(f"WSI dimensions: Width = {width}, Height = {height}")
    seg_level = 1

    # Run the segmentation and  tiling procedure
    start_time = time.time()
    tissue_mask_scaled = create_tissue_mask(wsi, seg_level)
    print('Get tissue mask scaled')
    filtered_tiles, names = create_tissue_tiles(wsi, tissue_mask_scaled, args.tile_size)
    print('Get filtered_tiles...')


    # Build a figure for quality control purposes, to check if the tiles are where we expect them.
    qc_img = make_tile_QC_fig(filtered_tiles, wsi, seg_level, 5, names=names,
                                  out_dir=os.path.join(args.output_dir, slide_id, 'tiles'))

    qc_img_target_width = 1920
    qc_img = qc_img.resize(
        (qc_img_target_width, int(qc_img.height / (qc_img.width / qc_img_target_width)))
    )
    print(
        f"Finished creating {len(filtered_tiles)} tissue tiles in {time.time() - start_time}s"
    )

    count_features = len(filtered_tiles)
    # Save QC figure while keeping track of number of features/tiles used since RBG filtering is within DataLoader.
    qc_img_file_path = os.path.join(
        args.output_dir, f"{slide_id}_w{width}_h{height}_features_QC.png"
    )
    qc_img.save(qc_img_file_path)
    print(
        f"Finished extracting {count_features} features in {(time.time() - start_time):.2f}s"
    )

