import os
import FastGeodis
import numpy as np
import SimpleITK as sitk
import torch
import warnings


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
    #changed intensity for tests, remove these first 3 lines for regular Geodis
    fastraster_output_gpu = np.where(input_image == -1024, -1024, fastraster_output_gpu) #background
    #probably remove these two
    fastraster_output_gpu = np.clip(fastraster_output_gpu, -900, None)
    fastraster_output_gpu = np.where(fastraster_output_gpu >= 0, 100, fastraster_output_gpu)

    fastraster_output_image = sitk.GetImageFromArray(fastraster_output_gpu)
    fastraster_output_image.SetSpacing(spacing_raw)
    fastraster_output_image.SetOrigin(image_sitk.GetOrigin())
    fastraster_output_image.SetDirection(image_sitk.GetDirection())


    try:
        elastix = sitk.ElastixImageFilter()
        elastix.SetFixedImage(image_sitk)
        elastix.SetMovingImage(fastraster_output_image)
        elastix.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
        elastix.LogToConsoleOff()
        elastix.Execute()
        fastraster_output_image_RO = elastix.GetResultImage()

    except RuntimeError as e:
        warnings.warn(str(e))

    return fastraster_output_image_RO

if __name__ == "__main__":
    ct_image = "D:\\Data\\CNH_Paired\\NoBedCTs\\377_noBed.nii.gz"
    seed_image = "D:\\Data\\CNH_Paired\\377_suture_seg.nii.gz"
    GD_CT_folder = "D:\\Data\\CNH_Paired"

    gd_CT_image = geodesic_distance3d_3dmasked(ct_image, seed_image)
    sitk.WriteImage(gd_CT_image, os.path.join(GD_CT_folder, "377_geodis.nii.gz"))

    print("Done!")