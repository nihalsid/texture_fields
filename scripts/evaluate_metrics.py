import mesh2tex.utils.FID.fid_score as FID
import mesh2tex.utils.FID.feature_l1 as feature_l1
import mesh2tex.utils.SSIM_L1.ssim_l1_score as SSIM


if __name__ == '__main__':
    gt = "/home/yawar/baseline_renders/gt_224"
    tf = "/home/yawar/baseline_renders/pifu_224/000"
    print(SSIM.calculate_ssim_l1_given_paths([gt, tf], subfolder_mode=True))
    print(feature_l1.calculate_feature_l1_given_paths([gt, tf], 16, True, 2048, subfolder_mode=True))
    print(FID.calculate_fid_given_paths([gt, tf], 16, True, 2048, subfolder_mode=True))
