addpath('metrics')
video_name = {'BasketballDril'};
frame_count = {596};

scale = 4;
degradation = 'RA';
psnr_avg = [];
ssim_avg = [];
results_name={'RA_QP22'};
for result=1:length(results_name)
    for idx_video = 1:length(video_name)
        psnr_video = [];
        ssim_video = [];
        for idx_frame = 3:frame_count{idx_video} 				% exclude the first and last 2 frames
            img_hr = imread(['/testing/dataset/path/GT', video_name{idx_video},'/', num2str(idx_frame,'%03d'),'.png']);
            img_sr = imread(['/sr/results/path/',video_name{idx_video},'/', num2str(idx_frame,'%03d'),'.png']);
            
            h = min(size(img_hr, 1), size(img_sr, 1));
            w = min(size(img_hr, 2), size(img_sr, 2));
            
            border = 10;

            img_hr_ycbcr = rgb2ycbcr(img_hr);
            %img_hr_y = img_hr_ycbcr;
            img_hr_y = img_hr_ycbcr(1+border:h-border, 1+border:w-border, 1);
            %img_sr_y = img_sr;
            img_sr_ycbcr = rgb2ycbcr(img_sr);
            img_sr_y = img_sr_ycbcr(1+border:h-border, 1+border:w-border, 1);
            %img_sr_y = img_sr_ycbcr;
            psnr_video(idx_frame) = cal_psnr(img_sr_y, img_hr_y);
            ssim_video(idx_frame) = cal_ssim(img_sr_y, img_hr_y);
        end
        psnr_avg(idx_video) = mean(psnr_video);
        ssim_avg(idx_video) = mean(ssim_video);
        disp([video_name{idx_video},'---Mean PSNR: ', num2str(mean(psnr_video),'%0.4f'),', Mean SSIM: ', num2str(mean(ssim_video),'%0.4f')])
    end
    disp(['---------------------------------------------'])
    disp([results_name{result}, degradation,'_x', num2str(scale) ,' SR---Mean PSNR: ', num2str(mean(psnr_avg),'%0.4f'),', Mean SSIM: ', num2str(mean(ssim_avg),'%0.4f')])
end
