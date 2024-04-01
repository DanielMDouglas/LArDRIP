%matplotlib inline
import matplotlib
#import matplotlib as mpl
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PyQt5
import random
#torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def find_smallest_track_center_and_volume(tracks):
    smallest_volume = np.inf
    smallest_track_center = None

    for track in tracks:
        if track.size == 0:  # Skip empty tracks
            continue

        min_coords = np.min(track, axis=0)
        max_coords = np.max(track, axis=0)
        volume = np.prod(max_coords - min_coords)
        if volume < smallest_volume:
            smallest_volume = volume
            smallest_track_center = (max_coords + min_coords) / 2  # Center of the track

    if smallest_track_center is None:
        raise ValueError("All tracks are empty or no tracks provided.")

    return smallest_track_center, smallest_volume

def load_data_and_create_incomplete_tracks(file_path, box_size):
    tracks = []
    missing_regions = []
    before_and_after_regions = []
    some_large_number = 50000  

    with h5py.File(file_path, 'r') as f:
        images_data = f['images']
        num_images = len(np.unique(images_data['imageInd']))
        count = 0
        for iteration in range(some_large_number):
            # Use modulo operation to cycle through image indices
            image_ind = iteration % num_images

            image_indices = np.where(images_data['imageInd'] == image_ind)
            vox_data = images_data[image_indices]

            voxx = vox_data['voxx']
            voxy = vox_data['voxy']
            voxz = vox_data['voxz']
            voxdE = vox_data['voxdE']

            track = np.vstack((voxx, voxy, voxz, voxdE)).T
            if len(track) == 726:
                tracks.append(track)
                print(count)
                count += 1

            # Break loop at 500 tracks
            if len(tracks) == 500:
                break

    # Normalize the data
    min_val = np.min(np.vstack(tracks), axis=0)
    max_val = np.max(np.vstack(tracks), axis=0)
    data_range = max_val - min_val
    normalized_tracks = [(track - min_val) / (max_val - min_val) for track in tracks]
    
    # Normalize the box_size
    print(np.array(box_size), data_range)
    normalized_box_size = np.array(box_size) / data_range[:3]
    smallest_track_center, _ = find_smallest_track_center_and_volume(normalized_tracks)

    
    new_x_min = smallest_track_center[0] - normalized_box_size[0] / 2
    new_x_max = smallest_track_center[0] + normalized_box_size[0] / 2

    count = 0
    for track in normalized_tracks:
        # Use new_x_min and new_x_max to determine inside and outside box points
        inside_box = (track[:, 0] >= new_x_min) & (track[:, 0] <= new_x_max)
        missing_region = track[inside_box]
        before_missing = track[track[:, 0] < new_x_min]
        after_missing = track[track[:, 0] > new_x_max]

        if missing_region.size == 0:
            continue

        concatenated_before_after = np.concatenate([before_missing, after_missing], axis=0)
        before_and_after_regions.append(concatenated_before_after)
        missing_regions.append(missing_region)
        
        if count % 50 == 0:
            print(f'Completed track {count}')
        count += 1

    return normalized_tracks, before_and_after_regions, missing_regions, new_x_min, new_x_max


file_path = '/path/to/h5_file.h5'
box_size = (5, 1000, 1000) 
normalized_tracks, _, _, _, _ = load_data_and_create_incomplete_tracks(file_path, box_size)
normalized_tracks = np.asarray(normalized_tracks)
normalized_tracks = torch.from_numpy(normalized_tracks).float().to(device)

save_path = r'.../normalized_tracks_np_mul.npy'

