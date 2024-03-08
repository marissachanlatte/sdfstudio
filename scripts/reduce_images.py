import os
import numpy as np
import shutil
import json

def main() -> None:
    dir_path = '/global/homes/m/mzweig/sdfstudio/data/cube'
    num_samples = 250
    num_images = len(os.listdir(dir_path)) - 1
    new_dir = dir_path + str(num_samples)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    random_samples = np.random.choice(num_images, size=num_samples, replace=False)
    for file in os.listdir(dir_path):
        if file[0] == '0':
            file_num = int(file.split('_')[0])
        if file_num in random_samples:
            shutil.copyfile(os.path.join(dir_path, file), os.path.join(new_dir, file))

    # Update JSON
    source = os.path.join(dir_path, 'meta_data.json')
    new_json = os.path.join(new_dir, 'meta_data.json')
    shutil.copyfile(source, new_json)

    with open(new_json, 'r') as f:
        data = json.load(f)
        frames = data['frames']
        new_frames = []

        for frame in frames:
            frame_id = int(frame['rgb_path'].split('_')[0])
            if frame_id in random_samples:
                new_frames.append(frame)
        data['frames'] = new_frames

    os.remove(new_json)
    with open(new_json, 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    main()