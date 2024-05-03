import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import joblib
import glob

all_stats = [0 for _ in range(10000)]
for i, path in enumerate(glob.glob(os.path.join("collapsed_data", "train", "*"))):
    img = cv2.imread(path)
    stats = {#"min_inten": np.min(img),
             #"max_inten": np.max(img),
             "ave_inten": np.mean(img),
             "pix_hgt": img.shape[0],
             "pix_wid": img.shape[1]}
    all_stats[i] = stats

tot_min_hgt, tot_min_wid = min([stats["pix_hgt"] for stats in all_stats]), min([stats["pix_wid"] for stats in all_stats])
tot_max_hgt, tot_max_wid = max([stats["pix_hgt"] for stats in all_stats]), max([stats["pix_wid"] for stats in all_stats])
tot_ave_hgt = sum([stats["pix_hgt"] for stats in all_stats]) / 10000.0
tot_ave_wid = sum([stats["pix_wid"] for stats in all_stats]) / 10000.0
tot_ave_inten = sum([stats["ave_inten"] for stats in all_stats]) / 10000.0

print("tot_min_hgt, tot_min_wid:", tot_min_hgt, tot_min_wid)
print("tot_max_hgt, tot_max_wid:", tot_max_hgt, tot_max_wid)
print("tot_ave_hgt:", tot_ave_hgt)
print("tot_ave_wid:", tot_ave_wid)
print("tot_ave_inten:", tot_ave_inten)

plt.hist([stats["ave_inten"] for stats in all_stats], bins=20, label="ave_inten")
plt.show()
plt.hist([stats["pix_hgt"] for stats in all_stats], bins=20, label="pix_hgt")
plt.show()
plt.hist([stats["pix_wid"] for stats in all_stats], bins=20, label="pix_wid")
plt.show()