import os
import FastGeodis
import numpy as np
import SimpleITK as sitk
import torch


def geodesic_distance3d_3dmasked(CT, seed_mask):
    image_sitk = sitk.ReadImage(CT)
    input_image = sitk.GetArrayFromImage(image_sitk)
    spacing_raw = image_sitk.GetSpacing()
    spacing = [spacing_raw[2], spacing_raw[1], spacing_raw[0]]
    input_image = np.asarray(input_image, np.float32)

    seed_image = sitk.ReadImage(seed_mask)
    seed_image = sitk.GetArrayFromImage(seed_image)
    seed_image = np.asarray(seed_image, np.float32)

    input_image_pt = torch.from_numpy(input_image).unsqueeze_(0).unsqueeze_(0)
    seed_image_pt = torch.from_numpy(seed_image).unsqueeze_(0).unsqueeze_(0)

    device = ("cuda" if input_image_pt.shape[1] == 1 and torch.cuda.is_available() else None)

    if device:
        input_image_pt = input_image_pt.to(device)
        seed_image_pt = seed_image_pt.to(device)
        fastraster_output_gpu = np.squeeze(
            FastGeodis.signed_generalised_geodesic3d(input_image_pt, seed_image_pt, spacing, 1e10, 1.0, 4)
            .detach().cpu().numpy())

    #compare two arrays and set the 0 value to -1024 for the geodisic case
    fastraster_output_gpu = np.where(input_image == -1024, -1024, fastraster_output_gpu)
    fastraster_output_image = sitk.GetImageFromArray(fastraster_output_gpu)
    fastraster_output_image.SetSpacing(spacing_raw)
    return fastraster_output_image

if __name__ == "__main__":
    image_folder = "D:\\Data\\CNH_Paired\\NoBedCTs"
    seed_folder = "D:\\Data\\CNH_Paired\\NoBedCTseeds"
    GD_CT_folder = "D:\\Data\\CNH_Paired\\GeoDis_CT"

    isExist = os.path.exists(GD_CT_folder)
    if not isExist:
        os.makedirs(GD_CT_folder)

    CTs_with_paths = [f.path for f in os.scandir(image_folder)]
    seed_files = [f for f in os.listdir(seed_folder)]
    seed_dict = {f.split("_")[0]: f for f in seed_files}

    for ct_filename in CTs_with_paths:
        seed_file = seed_dict.get(ct_filename.split("\\")[-1].split("_")[0])
        if seed_file:
            ct_image = os.path.join(image_folder, ct_filename)
            seed_image = os.path.join(seed_folder, seed_file)
            gd_CT_image = geodesic_distance3d_3dmasked(ct_image, seed_image)
            sitk.WriteImage(gd_CT_image, os.path.join(GD_CT_folder, seed_file.split("_")[0] + "_geodis.nii.gz"))

    print("Done!")