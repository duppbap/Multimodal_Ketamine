function [M_postmc] = MotionCorr(mat3D, showVid)
%MCorr piecewise rigid motion correction. output video comparing pre and
%post motion correction
%INPUT
% mat3D - raw fUSI matrix
% showVid - boolean for showing video or not
%OUTPUT
% M_postmc - mat3D after motion correction

rigidParam = NoRMCorreSetParms('d1', size(mat3D, 1),...
                                  'd2', size(mat3D, 2),...
                                  'bin_width', 120,...
                                  'max_shift', 10, ...
                                  'mot_uf', 50, ...
                                  'init_batch', 40);

%perform mc
[M_postmc, ~, ~, ~] = normcorre(mat3D, rigidParam);

%% compare pre and post mc 
%video 
if showVid
v = VideoWriter('side_by_side_comparison.avi');
open(v);
numFrames = size(mat3D, 3);
figure; set(gcf,'windowstyle', 'docked');
    for i = 1:numFrames
        subplot(1, 2, 1);
        imagesc(mat3D(:,:,i));
        colormap('gray');
        colorbar;
        clim([0 100]);
        axis off;
        title('Original Data');

        subplot(1, 2, 2);
        imagesc(M_postmc(:,:,i));
        colormap('gray');
        colorbar;
        clim([0 100]);
        axis off;
        title('Motion Corrected Data');

        frame = getframe(gcf);
        writeVideo(v, frame);
    end
    close(v);
end

%% standard deviation map

% Calculate standard deviation across the time dimension
std_pre = std(mat3D, 0, 3); % Original data
std_post = std(M_postmc, 0, 3); % Motion-corrected data

% Plot the standard deviation maps
% figure; 
% subplot(1, 2, 1);
% imagesc(std_pre);
% colormap('hot');
% colorbar
% title('Standard Deviation (Original)');
% axis off;
% axis square;
% set(gca,'fontsize',30);
% 
% subplot(1, 2, 2);
% imagesc(std_post);
% colormap('hot');
% colorbar;
% title('Standard Deviation (Motion Corrected)');
% axis off;
% axis square;
% set(gca,'fontsize',30);


end