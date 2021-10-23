import sys
import numpy as np
from torchvision.transforms import Resize
import csv
from PIL import Image
from tqdm.auto import tqdm


imagenet_path = sys.argv[1]
target_path = sys.argv[2]
print(f"Generate numpy datset from {imagenet_path} to {target_path}")

transform = Resize(100)
def convert(mode):
    data_dict = {}
    csvf = imagenet_path + f"/{mode}.csv"
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in tqdm(enumerate(csvreader)):
            filename = row[0]
            label = row[1]
            if(label not in data_dict):
                data_dict[label] = []
            
            path = f"{imagenet_path}/images/{filename}"
            img = Image.open(path).convert('RGB')
            img = transform(img)
            arr = np.asarray(img)
            data_dict[label].append(arr)
    return data_dict
for mode in ["val", "test", "train"]:
    data_dict = convert(mode)
    np.save(target_path+f"/{mode}.npy", data_dict)
    print (f"{mode} data was generated successfully")

