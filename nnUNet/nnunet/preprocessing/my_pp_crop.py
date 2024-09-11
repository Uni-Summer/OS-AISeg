import os
import pickle
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
from matplotlib import pyplot as plt

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    # fig, ax = plt.subplots(1,8)
    # for i in range(8):
    #     ax[i].imshow(nonzero_mask[i])
    # plt.show()
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask, 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    if seg is not None:
        cropped_seg = []
        for c in range(seg.shape[0]):
            cropped = crop_to_bbox(seg[c], bbox)
            cropped_seg.append(cropped[None])
        seg = np.vstack(cropped_seg)

    nonzero_mask = crop_to_bbox(nonzero_mask, bbox)[None]
    if seg is not None:
        seg[(seg == 0) & (nonzero_mask == 0)] = nonzero_label
    else:
        nonzero_mask = nonzero_mask.astype(int)
        nonzero_mask[nonzero_mask == 0] = nonzero_label
        nonzero_mask[nonzero_mask > 0] = 0
        seg = nonzero_mask
    return data, seg, bbox


def load_case_from_list_of_files(data_files, seg_file=None):
    assert isinstance(data_files, list) or isinstance(data_files, tuple), "case must be either a list or a tuple"
    properties = OrderedDict()
    data_itk = [sitk.ReadImage(f) for f in data_files]

    properties["original_size_of_raw_data"] = np.array(data_itk[0].GetSize())[[2, 1, 0]]
    properties["original_spacing"] = np.array(data_itk[0].GetSpacing())[[2, 1, 0]]
    properties["list_of_data_files"] = data_files
    properties["seg_file"] = seg_file

    properties["itk_origin"] = data_itk[0].GetOrigin()
    properties["itk_spacing"] = data_itk[0].GetSpacing()
    properties["itk_direction"] = data_itk[0].GetDirection()

    data_npy = np.vstack([sitk.GetArrayFromImage(d)[None] for d in data_itk])
    if seg_file is not None:
        seg_itk = sitk.ReadImage(seg_file)
        seg_npy = sitk.GetArrayFromImage(seg_itk)[None].astype(np.float32)
    else:
        seg_npy = None
    return data_npy.astype(np.float32), seg_npy, properties



def crop(data, properties, seg=None):
        shape_before = data.shape
        data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
        shape_after = data.shape
        print("before crop:", shape_before, "after crop:", shape_after)

        properties["crop_bbox"] = bbox
        properties['classes'] = np.unique(seg)
        seg[seg < -1] = 0
        properties["size_after_cropping"] = data[0].shape
        return data, seg, properties


def crop_from_list_of_files(data_files, seg_file=None):
        data, seg, properties = load_case_from_list_of_files(data_files, seg_file)
        return crop(data, properties, seg)

if __name__ == '__main__':
    data_files = ['/media/summer/新加卷1/dataset/ISLES2018/HU_processed/case_1/SMIR.Brain.XX.O.CT.339203.nii']
    seg_file = '/media/summer/新加卷1/dataset/ISLES2018/HU_processed/case_1/SMIR.Brain.XX.O.OT.339208.nii'
    # data_files = ['/home/summer/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task800_Stroke/imagesTr/stroke_001_0000.nii.gz']
    # seg_file = '/home/summer/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task800_Stroke/labelsTr/stroke_001.nii.gz'
    case_identifier = 'case_1'
    out_folder = '/media/summer/新加卷1/dataset/ISLES2018/nnunet_pp_crop/' + case_identifier
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    data, seg, properties = crop_from_list_of_files(data_files, seg_file)

    all_data = np.vstack((data, seg))
    np.savez_compressed(os.path.join(out_folder, "%s.npz" % case_identifier), data=all_data)
    with open(os.path.join(out_folder, "%s.pkl" % case_identifier), 'wb') as f:
        pickle.dump(properties, f)
    
    with open('/home/summer/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_cropped_data/Task800_Stroke/dataset_properties.pkl', 'rb') as f:
        dataset_properties = pickle.load(f)
    img = data[0]
    intensityproperties = dataset_properties['intensityproperties']
    mean_intensity = intensityproperties[0]['mean']
    std_intensity = intensityproperties[0]['sd']
    lower_bound = intensityproperties[0]['percentile_00_5']
    upper_bound = intensityproperties[0]['percentile_99_5']
    HU_img = np.clip(img, lower_bound, upper_bound)
    z_s_img = (HU_img - mean_intensity) / std_intensity
    num = 0
    fig, ax = plt.subplots(2, 8)
    for i in range(2):
        for j in range(8):
            if num < 8:
                ax[i,j].imshow(HU_img[num])
            else:
                ax[i,j].imshow(z_s_img[num-8])
            num = num + 1
    plt.show()
    print('222222222222222222')