import math
import os
import time
from functools import wraps

import FastGeodis
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch


def demo_geodesic_distance3d(image_path, seed_pos):
    image_folder = os.path.dirname(image_path)
    image_sitk = sitk.ReadImage(image_path)
    input_image = sitk.GetArrayFromImage(image_sitk)
    spacing_raw = image_sitk.GetSpacing()
    spacing = [spacing_raw[2], spacing_raw[1], spacing_raw[0]]

    input_image = np.asarray(input_image, np.float32)
    #seed_image = np.zeros_like(input_image, np.uint8)
    #seed_image[seed_pos[0]][seed_pos[1]][seed_pos[2]] = 1
    seed_image = sitk.ReadImage(seed_pos)
    seed_image = sitk.GetArrayFromImage(seed_image)
    seed_image = np.asarray(seed_image, np.float32)
    device = "cpu"
    input_image_pt = torch.from_numpy(input_image).unsqueeze_(0).unsqueeze_(0)
    #seed_image_pt = (torch.from_numpy(1 - seed_image.astype(np.float32)).unsqueeze_(0).unsqueeze_(0))
    seed_image_pt = torch.from_numpy(seed_image).unsqueeze_(0).unsqueeze_(0)
    input_image_pt = input_image_pt.to(device)
    seed_image_pt = seed_image_pt.to(device)

    device = (
        "cuda" if input_image_pt.shape[1] == 1 and torch.cuda.is_available() else None
    )
    if device:
        input_image_pt = input_image_pt.to(device)
        seed_image_pt = seed_image_pt.to(device)
        tic = time.time()
        fastraster_output_gpu = np.squeeze(
            FastGeodis.signed_generalised_geodesic3d(
                input_image_pt, seed_image_pt, spacing, 1e10, 1.0, 4
            )
            .detach()
            .cpu()
            .numpy()
        )
        fastraster_time_gpu = time.time() - tic


    if device:
        print("FastGeodis GPU raster: {:.6f} s".format(fastraster_time_gpu))

    #compare two arrays and set the -1024 value to 0 in the geodisic case
        input_image_sub = sitk.GetImageFromArray(input_image)

    input_image_sub.SetSpacing(spacing_raw)
    sitk.WriteImage(input_image_sub, os.path.join(image_folder, "image3d_sub_2bed.nii.gz"))

    input_image_sub = sitk.GetArrayFromImage(input_image_sub)
    fastraster_output_gpu = np.where(input_image_sub == -1024, -1024, fastraster_output_gpu)
    fastraster_output_image = sitk.GetImageFromArray(fastraster_output_gpu)
    fastraster_output_image.SetSpacing(spacing_raw)
    sitk.WriteImage(fastraster_output_image, os.path.join(image_folder, "Geodis3D_2bed.nii.gz"))

    '''input_image = input_image * 255 / input_image.max()
    input_image = np.asarray(input_image, np.uint8)

    image_slice = input_image[101]
    if device:
        fastraster_output_gpu_slice = fastraster_output_gpu[101]

    plt.figure(figsize=(18, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(image_slice, cmap="gray")
    plt.autoscale(False)
    plt.plot([250], [290], "ro")
    plt.axis("off")
    plt.title("(a) Input image")

    if device:
        plt.subplot(2, 2, 2)
        plt.imshow(fastraster_output_gpu_slice)
        plt.axis("off")
        plt.title("(b) FastGeodis (gpu) | ({:.4f} s)".format(fastraster_time_gpu))'''

    print("Done!")
    #plt.show()

if __name__ == "__main__":
    demo_geodesic_distance3d("C:\\Users\\Austin Tapp\\Desktop\\33_noBed.nii.gz", "C:\\Users\\Austin Tapp\\Desktop\\seg.nii.gz")

#demo - ITK: 70, 60, 10 --> 10, 60, 70
#17 no bed, ON SUTURE - ITK: 250, 290, 102 --> 101, 290, 250
#17 no bed, ON SKULL - ITK: 260, 65, 14 --> 13, 65, 260
#17 no bed, on big soft spot - ITK: 246, 96, 97 --> 96, 96, 246
#makes sutures very dark, invert image intensities?
    #seems like a good option to me.