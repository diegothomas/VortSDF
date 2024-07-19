from multiprocessing import Pool
import cv2
import numpy as np
import math

def computePSNRArrays(gtImg, maskImg, testImg, diffDst = None):
    gtImg = gtImg[:,:,:3]
    testImg = testImg[:,:,:3]
    maskImg = maskImg[:,:,:3]

    if testImg.shape[:2] != gtImg.shape[:2]:
        testImg = cv2.resize(testImg, (gtImg.shape[1], gtImg.shape[0]))

    assert(testImg.shape[:2] == maskImg.shape[:2])

    maskImg = maskImg.astype(np.float64) / 255.0
    testImg = testImg.astype(np.float64) / 255.0
    gtImg = gtImg.astype(np.float64) / 255.0

    testImg *= maskImg
    gtImg *= maskImg

    if diffDst != None:
        cv2.imwrite(diffDst, (255*np.abs(testImg - gtImg)).astype(np.uint8))
    # cv2.imshow('diff', np.abs(testImg - gtImg))
    # cv2.waitKey(0)

    nPixels = maskImg.sum() # The number of channels is taken into account there
    if nPixels > 0:
        mse = np.mean((testImg - gtImg)**2) * (gtImg.shape[0] * gtImg.shape[1]) / nPixels
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))
        return psnr
    else:
        return np.nan


def computePSNROnce(image_path, ground_truth_path, mask_path):

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    image = image[:,:,:3]
    ground_truth = ground_truth[:,:,:3]

    if image.shape[:2] != ground_truth.shape[:2]:
        image = cv2.resize(image, ground_truth.shape[:2])

    assert(ground_truth.shape[:2] == mask.shape[:2])

    # cv2.imshow('mask', mask)
    # cv2.imshow('image', image)
    # cv2.imshow('ground_truth', ground_truth)
    # cv2.waitKey(0)

    mask = mask.astype(np.float64) / 255.0
    image = image.astype(np.float64) / 255.0
    ground_truth = ground_truth.astype(np.float64) / 255.0

    image *= mask
    ground_truth *= mask

    nPixels = mask.sum()
    if nPixels > 0:
        mse = np.mean((image - ground_truth)**2) * (ground_truth.shape[0] * ground_truth.shape[1]) / nPixels
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))
        return psnr
    else:
        return np.nan


def computePSNR(images, ground_truths, masks):
    # res = [computePSNROnce(*r) for r in zip(images, ground_truths, masks)]
    p = Pool()
    res = p.starmap(computePSNROnce, [r for r in zip(images, ground_truths, masks)])
    res = np.array(res)
    res = res[~np.isnan(res)] # remove nan values due to empty cameras

    return np.mean(res)


if __name__ == "__main__":
    test_cameras_idx = [14, 25, 31, 44, 58, 64, 65, 71, 85, 98, 104, 115]


    images = []
    ground_truths = []
    masks = []

    frame = 242

    for i in test_cameras_idx:
        images.append(f"/disk/btoussai/humanrf_dataset/Actor05/Sequence2/4x/LOD0/renders/img_{i-1}.png")
        ground_truths.append(f"/disk/btoussai/humanrf_dataset/Actor05/Sequence2/4x/rgbs/Cam{i:03d}/Cam{i:03d}_rgb{frame:06d}.jpg")
        masks.append(f"/disk/btoussai/humanrf_dataset/Actor05/Sequence2/4x/masks/Cam{i:03d}/Cam{i:03d}_mask{frame:06d}.png")

    computePSNR(images, ground_truths, masks)