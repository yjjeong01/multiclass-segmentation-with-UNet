#!/usr/bin/python3

import argparse
import glob
import os

import cv2
import numpy as np

from extract_event_and_frame_times import (get_sequence_name,
                                           get_camera_name,
                                           get_category,
                                           get_purpose)

IMG_HEIGHT = int(240 * 1.25)
IMG_WIDTH = int(320 * 1.25)


# From pydvs evimo-gen.py
def mask_to_color(mask):
    colors = [[84, 71, 140], [44, 105, 154], [4, 139, 168],
              [13, 179, 158], [22, 219, 147], [131, 227, 119],
              [185, 231, 105], [239, 234, 90], [241, 196, 83],
              [242, 158, 76], [239, 71, 111], [255, 209, 102],
              [6, 214, 160], [17, 138, 178], [7, 59, 76],
              [6, 123, 194], [132, 188, 218], [236, 195, 11],
              [243, 119, 72], [213, 96, 98]]

    cmb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    m_ = np.max(mask) + 500
    m_ = max(m_, 3500)

    maxoid = int(m_ / 1000)
    for i in range(maxoid):
        cutoff_lo = 1000.0 * (i + 1.0) - 5
        cutoff_hi = 1000.0 * (i + 1.0) + 5
        cmb[np.where(np.logical_and(mask >= cutoff_lo, mask <= cutoff_hi))] = np.array(colors[i % len(colors)])

    return cmb


def get_frame_by_index(frames, index):
    frames_names = list(frames.keys())
    frames_names.sort()
    frame_name = frames_names[index]

    return np.copy(frames[frame_name])  # To extract and keep in RAM


def on_trackbar(t_ms, scene_name):
    t = (t_ms / 1000.0) + t_start
    # Make images for flea3_7 if in sequence else black
    if flea3_7_classical_timestamps is not None:
        # Black if there is no data at this time
        if flea3_7_classical_timestamps[0] <= t <= flea3_7_classical_timestamps[-1] + 1 / 30.0:
            i_classical = np.searchsorted(flea3_7_classical_timestamps, t) - 1
            flea3_7_img = get_frame_by_index(flea3_7_data, i_classical)
            flea3_7_img_bgr = cv2.resize(flea3_7_img, dsize=(IMG_WIDTH, IMG_HEIGHT))
            cv2.imwrite(fr"../dataset/{scene_name}/img/{t}.png", flea3_7_img_bgr)

            i_depth = np.searchsorted(flea3_7_depth_timestamps, t) - 1
            flea3_7_m = get_frame_by_index(flea3_7_mask, i_depth)
            flea3_7_m = cv2.resize(flea3_7_m, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(
                np.float32)
            col_mask = mask_to_color(flea3_7_m)
            col_mask = cv2.cvtColor(col_mask, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(fr"../dataset/{scene_name}/mask/{t}.png", col_mask)


def group_files_by_sequence_name(files):
    files_grouped = {}

    for file in files:
        sequence_name = get_sequence_name(file)
        camera_name = get_camera_name(file)
        category = get_category(file)
        purpose = get_purpose(file)

        if sequence_name in files_grouped:
            files_grouped[sequence_name][3][camera_name] = file
        else:
            files_grouped[sequence_name] = [category, purpose, sequence_name, {camera_name: file}]

    return files_grouped


# Load an EVIMO2_v2 npz format list of events into RAM
def load_events(folder):
    events_t = np.load(os.path.join(folder, 'dataset_events_t.npy'), mmap_mode='r')
    events_xy = np.load(os.path.join(folder, 'dataset_events_xy.npy'), mmap_mode='r')
    events_p = np.load(os.path.join(folder, 'dataset_events_p.npy'), mmap_mode='r')

    events_t = np.atleast_2d(events_t.astype(np.float32)).transpose()
    events_p = np.atleast_2d(events_p.astype(np.float32)).transpose()

    events = np.hstack((events_t, events_xy.astype(np.float32), events_p))

    return events


list_scene = os.listdir(r"../../evimo/flea3_7/sfm/train")
list_scene += os.listdir(r"../../evimo/flea3_7/sfm/eval")


def main(scene_name):
    parser = argparse.ArgumentParser(
        description='View all cameras of a sequence with GT depth overlaid to see availability')
    parser = argparse.ArgumentParser()
    parser.add_argument('--idir', default=r"../../evimo", type=str,
                        help='Directory containing npz file tree')
    parser.add_argument('--seq', default=scene_name, help='Sequence name')
    args = parser.parse_args()

    data_base_folder = args.idir
    file_glob = data_base_folder + '/*/*/*/*'
    files = sorted(list(glob.glob(file_glob)))

    files_grouped_by_sequence = group_files_by_sequence_name(files)
    folders = files_grouped_by_sequence[args.seq][3]

    print('Opening npy files,', scene_name)

    if 'flea3_7' in folders:
        global flea3_7_data
        global flea3_7_mask
        global flea3_7_depth_timestamps
        global flea3_7_classical_timestamps

        flea3_7_data = np.load(os.path.join(folders['flea3_7'], 'dataset_classical.npz'))
        flea3_7_mask = np.load(os.path.join(folders['flea3_7'], 'dataset_mask.npz'))
        meta = np.load(os.path.join(folders['flea3_7'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        frame_infos = meta['frames']

        flea3_7_depth_timestamps = []
        flea3_7_classical_timestamps = []
        for frame in frame_infos:
            if 'gt_frame' in frame:
                flea3_7_depth_timestamps.append(frame['ts'])
            if 'classical_frame' in frame:
                flea3_7_classical_timestamps.append(frame['ts'])
        flea3_7_depth_timestamps = np.array(flea3_7_depth_timestamps)
        flea3_7_classical_timestamps = np.array(flea3_7_classical_timestamps)
    else:
        flea3_7_data = None
        flea3_7_mask = None
        flea3_7_depth_timestamps = None
        flea3_7_classical_timestamps = None

    if 'left_camera' in folders:
        global left_camera_events
        global left_camera_mask
        global left_camera_timestamps

        left_camera_events = load_events(folders['left_camera'])
        left_camera_mask = np.load(os.path.join(folders['left_camera'], 'dataset_mask.npz'))
        meta = np.load(os.path.join(folders['left_camera'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        frame_infos = meta['frames']

        left_camera_timestamps = []
        for frame in frame_infos:
            if 'gt_frame' in frame:
                left_camera_timestamps.append(frame['ts'])
        left_camera_timestamps = np.array(left_camera_timestamps)
    else:
        left_camera_events = None
        left_camera_mask = None
        left_camera_timestamps = None

    if 'right_camera' in folders:
        global right_camera_events
        global right_camera_mask
        global right_camera_timestamps

        right_camera_events = load_events(folders['right_camera'])
        right_camera_mask = np.load(os.path.join(folders['right_camera'], 'dataset_mask.npz'))
        meta = np.load(os.path.join(folders['right_camera'], 'dataset_info.npz'), allow_pickle=True)['meta'].item()
        frame_infos = meta['frames']

        right_camera_timestamps = []
        for frame in frame_infos:
            if 'gt_frame' in frame:
                right_camera_timestamps.append(frame['ts'])
        right_camera_timestamps = np.array(right_camera_timestamps)
    else:
        right_camera_events = None
        right_camera_mask = None
        right_camera_timestamps = None

    # Can't just use a max because of all the None's
    global t_start
    global t_end
    t_start = None
    t_end = None
    for timestamps in (flea3_7_depth_timestamps, flea3_7_classical_timestamps, left_camera_timestamps,
                       right_camera_timestamps, left_camera_events, right_camera_events):
        if timestamps is not None:
            if len(timestamps.shape) == 1:
                new_t_start = timestamps[0]
                new_t_end = timestamps[-1]
            else:
                new_t_start = timestamps[0, 0]
                new_t_end = timestamps[-1, 0]

            if t_start is None or new_t_start < t_start:
                t_start = new_t_start

            if t_end is None or new_t_end > t_end:
                t_end = new_t_end

    slider_max = int(1000 * (t_end - t_start))

    for i in range(1, slider_max, 100):
        on_trackbar(i, scene_name)


for scene in list_scene:
    os.makedirs(fr"../dataset/{scene}", exist_ok=True)
    os.makedirs(fr"../dataset/{scene}/img", exist_ok=True)
    os.makedirs(fr"../dataset/{scene}/mask", exist_ok=True)
    main(scene)
