import cv2
import numpy as np
from PIL import Image
import sys
import glob
import os

def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    # img1 = reorder_image(img1, input_order=input_order)
    # img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # if test_y_channel:
    #     img1 = to_y_channel(img1)
    #     img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    # img1 = reorder_image(img1, input_order=input_order)
    # img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # if test_y_channel:
    #     img1 = to_y_channel(img1)
    #     img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()

def main():
    # res_vid_name = [
    #         'BasketballDrive_fps50_480x272_500F.yuv', 
    #         'Kimono1_fps24_480x272_240F.yuv', 
    #         'BQTerrace_fps60_480x272_600F.yuv', 
    #         'ParkScene_fps24_480x272_240F.yuv'
    #         ]
    # gt_vid_name = [
    #            'BasketballDrive_1920x1080_50_500F.yuv', 
    #            'Kimono1_1920x1080_24_240F.yuv', 
    #            'BQTerrace_1920x1080_60_600F.yuv', 
    #            'ParkScene_1920x1080_24_240F.yuv'
    #            ]
    # cal_psnr_ssim('/data/cpl/testing_results/train_1126_LD_QP32_J_EDVR_M/', res_vid_name, gt_vid_name)
    pass

def cal_psnr_ssim(lr_path, gt_path):

    psnr = 0
    ssim = 0
    frames = 0
    for i in sorted(glob.glob(lr_path)):
        frames = frames + 1
        _, name = os.path.split(i)

        gt_img = os.path.join(gt_path,name)
        lr_img = os.path.join(lr_path,name)

        res_img = Image.open(lr_img)
        gt_img = Image.open(gt_img).convert('L')
        min_width = min(res_img.width, gt_img.width)
        min_height = min(res_img.height, gt_img.height)

        res_img = np.array(res_img)
        res_img = res_img[:min_height,:min_width,np.newaxis].astype(np.float64)
        gt_img = np.array(gt_img)
        gt_img = gt_img[:min_height,:min_width,np.newaxis].astype(np.float64)

        f_psnr = calculate_psnr(res_img, gt_img, 4, test_y_channel=True)
        f_ssim = calculate_ssim(res_img, gt_img, 4, test_y_channel=True)

        psnr += f_psnr
        ssim += f_ssim

        print('seq: %d (%.5f/.5f) ... ' % (frames, f_psnr, f_ssim), end='\r')


    msg = 'Average PSNR/SSIM: %.3f/%.5f' % ( psnr / frames, ssim / frames)
    print(msg)
    # seq_ave_msg = 'Sequence Average PSNR/SSIM: %.3f/%.5f' % (seq_ave_psnr / len(gt_vid_name), seq_ave_ssim / len(gt_vid_name))

if __name__ == '__main__':
    cal_psnr_ssim()
