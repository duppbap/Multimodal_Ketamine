%% mainScript
% Author: Wooseong Choi
% Summary: main script for Multimodal-Ketamine-TMS Study

%% intialize work space
clc; clear; close all;
data_path = '/project/wooseong_1271/Multimodal_Ketamine_Data';
addpath(genpath(data_path));
addpath(genpath('/home1/wooseong/Multimodal_Ketamine'));

%% set up local parallel pool
pc = parallel.cluster.Local;
job_folder = fullfile('/scratch1/',getenv('USER'),getenv('SLURM_JOB_ID'));
mkdir(job_folder);
set(pc,'JobStorageLocation',job_folder);
ncores = str2num(getenv('SLURM_CPUS_PER_TASK')) - 1;
pool = parpool(pc,ncores);

%% initalize fUSI data
allRats = {'rat10', 'rat13', 'rat14'};

% all samples
sample_idx = struct;

% M1 , S1 , thalamus, random, random
sample_idx.rat10 = [16 87; 8 62; 59 46 ];
sample_idx.rat13 = [18 110; 8 62; 39 37; 79 25; 84 96];
sample_idx.rat14 = [18 24; 6 77; 51 82];

TMS_onset.rat10 = 2300;
TMS_onset.rat13 = 2300;
TMS_onset.rat14 = 1800;

% specify samples
sampleName = allRats{3};
idx = sample_idx.(sampleName);
filename =[sampleName '.scan'];

%update data path
data_path = [data_path '/' allRats{2} '/figures'];

%% preprocess
mat4D = h5read(filename, '/Data');
mat3D = permute(squeeze(mat4D), [2 1 3]);
filt_params = struct('fs', 1, 'fc1', 0.01, 'fc2', 0.3,...
                     'filtType','low', 'order',4,...
                     'filtOpt', 0);
norm_params = struct('duration',720);
Mpreproc = preprocFUSI(mat3D, filt_params, norm_params); 
Mpreproc = Mpreproc(:,:,1:TMS_onset.(sampleName)); % cut after TMS

%gather dim
dim = struct;
[dim.y, dim.x, dim.t] = size(Mpreproc);
numVoxels = dim.y * dim.x;

% clear raw data for memory
clear mat4D; clear mat3D;


%% prepare PPS vector

% Read in PPS time stamps
fileName = [sampleName '.xlsx'];

pps = readmatrix(fileName);             
pps = floor(pps(pps<dim.t));
pps = reshape(pps, 2, [])';

% initialize
binaryEvent1 = zeros(1, size(Mpreproc,3));  
binaryEvent2 = zeros(1, size(Mpreproc,3));  
% binaryEvent3 = zeros(1, size(Mpreproc2,3));  

% single event and duration representations
for i = 1:size(pps, 1)
    t1 = pps(i, 1);
    t2 = pps(i, 2);
    binaryEvent1(t1) = 1;
    binaryEvent2(t1:t2) = 1;
end

% % each PS representation
% interpl_pps = pps * 10;
% for i = 1:size(pps,1)
%     t1 = interpl_pps(i,1);
%     t2 = interpl_pps(i,2);
%     binaryEvent3(t1:4:t2) = 1;
% end

%% plot fusi signal
figure; 
pbaspect([3 1 1]);

v = squeeze(Mpreproc(idx(2,1),idx(2,2),:));
plot(v,'b','LineWidth', 2); hold on;
title('S1 voxel', 'fontsize', 34, 'fontweight', 'b');
ylim([-3 100]);

% %plot pps events
for i = 1:size(pps,1)
    t1 = pps(i,1);
    t2 = pps(i,2);
    y_limits = ylim;
    patch([t1 t2 t2 t1],...
        [y_limits(1) y_limits(1) y_limits(2) y_limits(2)],...
        'k', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
end
xlim([0 TMS_onset.(sampleName)]);
legend('fUSI', 'PPS events');
ylabel('Z-score');
xlabel('Seconds');
legend('fUSI', 'PPS events');

% save 
saveas(gcf, fullfile(data_path, 's1_representative_signal.fig'));
   
%% RLS 

lambdas = [0.8 0.9 0.95 0.98 0.99 0.999];
m = 80;                             
tic;
rls = optRLS(Mpreproc, binaryEvent1, m, lambdas);
toc;

%% compare NMSE for each lambda for a representative voxel
[linear_idx] = sub2ind([dim.y dim.x], idx(2,1), idx(2,2));

% NMSE for S1
rls_nmse = rls.recursiveNMSE(:,linear_idx);
labels = cell(1,length(lambdas));
figure; 
for i = 1:length(lambdas)
    plot(rls_nmse{i}, 'LineWidth', 2);
    hold on;
    labels{i} = sprintf('λ = %.4f', lambdas(i));
end
xlim([1 dim.t]);
xlabel('Seconds');
ylabel('NMSE');
set(gca,'fontsize',18);

% save 
saveas(gcf, fullfile(data_path, 's1_representative_nmse.fig'));

%% extract the optimal lambda for each voxel

numLambdas = length(lambdas);

% initialize
finalNMSE = NaN(numLambdas, numVoxels);

% Extract the last NMSE value from each cell (assuming time-series data)
for i = 1:numLambdas
    for v = 1:numVoxels
        nmseVector = rls.recursiveNMSE{i, v};  
        finalNMSE(i, v) = nmseVector(end);  
    end
end

% Find the index of the minimum NMSE for each voxel
[~, optimalLambdaIdx] = min(finalNMSE, [], 1); % Min across lambda dimension

% Convert index to corresponding lambda values
optimalLambdas = lambdas(optimalLambdaIdx);

% convert to lambda map
lambdaMap = reshape(optimalLambdaIdx, dim.y, dim.x, []);
figure; 
imagesc(lambdaMap); 
cb = colorbar; 
cb.Ticks = 1:numLambdas;
cb.TickLabels = arrayfun(@(x) sprintf('%.4f', x), lambdas, 'UniformOutput', false);
ylabel(cb, 'Optimal Lambda');
yticks([]); xticks([]);
title('Lambda Map');
set(gca,'fontsize',18);
saveas(gcf, fullfile(data_path, 'lambda_map.fig'));

% histogram
figure; 
histogram(optimalLambdaIdx, 'BinEdges', 0.5:(numLambdas + 0.5));
xticks(1:numLambdas);
xticklabels(arrayfun(@(x) sprintf('%.4f', x), lambdas, 'UniformOutput', false));
xlabel('Optimal Lambda');
ylabel('Number of Voxels');
title('Optimal Lambda Distribution');
set(gca,'fontsize',18);
saveas(gcf, fullfile(data_path, 'lambda_histogram.fig'));

%% plot fit for representative voxel
M2D = double(reshape(Mpreproc, numVoxels, dim.t ));

%Retrieve data from representative voxel
linear_idx = sub2ind([dim.y dim.x], idx(2,1), idx(2,2)); 
y = M2D(linear_idx,:);
optLIdx = optimalLambdaIdx(linear_idx);
optL = optimalLambdas(linear_idx);
yhat = rls.all_y{optLIdx,linear_idx}';
khat = rls.all_k{optLIdx,linear_idx};
residuals = y - yhat;

% ---Visualization---
% yhat vs yobserved (overlay)
figure; 
plot(1:dim.t, y, 'k', 'LineWidth', 1); hold on;
plot(1:dim.t, yhat, 'm', 'LineWidth', 2);
plot(1:dim.t, khat(1,:),'c', 'LineWidth', 2);
legend('Observed', 'Predicted','Bias');
title(sprintf('Observed vs. Predicted (Lambda: %f)', optL));
xlabel('Seconds'); ylabel('Amplitude');
xlim([0 dim.t]);
set(gca,'fontsize',20);
hold off;
saveas(gcf, fullfile(data_path, 'S1_fit.fig'));

% Scatter plot yhat vs yobserved with red linear fit
figure; 
scatter(yhat, y, 'k'); hold on;
p = polyfit(yhat, y, 1);
yfit = polyval(p,yhat);
plot(yhat, yfit, 'r', 'LineWidth', 2);
title('Observed vs. Predicted');
xlabel('Predicted yhat'); ylabel('Observed y');
hold off;
set(gca,'fontsize',20);
axis square;
saveas(gcf, fullfile(data_path, 'S1_scatter1.fig'));

% Time-varying bias term (k0)
figure; 
plot(1:dim.t, khat(1,:),'k', 'LineWidth', 2);
title('Time-Varying Bias Term (k0)');
xlabel('Seconds'); ylabel('Bias Amplitude');
set(gca,'fontsize',20);
xlim([0 dim.t]);

% 3D Time-Varying Kernels (Perspective View)
% half hanning window for temporal filter
winLength = 10;
full_hann = hann(2 * winLength + 1);
half_hann = full_hann(1:winLength+1);
half_hann = half_hann / sum(half_hann);
half_hann = half_hann';
khat = convn(khat, half_hann, 'same');

% Build matching meshgrid
[m, dim.t] = size(khat);
[T, K] = meshgrid(1:dim.t, 1:m);
figure;
surf(T, K, khat, 'EdgeColor', 'interp');
xlabel('Seconds');
ylabel('Kernel Coefficients');
ylim([1 m]);
zlabel('Kernel Amplitude');
title('Time-Varying Kernels');
axis square;
view([-30, 60]);
shading interp;
colormap parula; colorbar;
set(gca,'fontsize',18);
saveas(gcf, fullfile(data_path, '3d_kernel.fig'));

% spatial filter
spatial_size = 3;
hanning_1D = hann(spatial_size);
hanning_3D = hanning_1D .* permute(hanning_1D, [2 1]) .* permute(hanning_1D, [3 2 1]);
hanning_3D = hanning_3D / sum(hanning_3D(:));

% heat map
figure; 
K_nobias = khat(2:end, :);
K_nobias = convn(K_nobias, hanning_3D, 'same');
imagesc(K_nobias); 
cb = colorbar;
colormap('parula');
ylabel(cb, 'Kernel Amplitude (Z-Score)');
xlabel('Seconds');
ylabel('Kernel Coefficients');
set(gca,'fontsize',18);
title('Time Varying Kernel');
%clim([0 4]);
saveas(gcf, fullfile(data_path, 'heatmap_kernel.fig'));

% Residuals vs. Predicted yhat
figure; 
scatter(yhat, residuals, 'k');
title('Residuals vs. Predicted');
xlabel('Predicted'); ylabel('Residuals');
axis square;
set(gca,'fontsize',20);
saveas(gcf, fullfile(data_path, 'scatter2.fig'));
