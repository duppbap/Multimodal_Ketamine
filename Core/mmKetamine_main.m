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
sampleName = allRats{2};
idx = sample_idx.(sampleName);
filename =[sampleName '.scan'];

%update data path
data_path = [data_path '/' sampleName '/figures'];

%% preprocess
mat4D = h5read(filename, '/Data');
mat3D = permute(squeeze(mat4D), [2 1 3]);
filt_params = struct('fs', 1, 'fc1', 0.01, 'fc2', 0.3,...
                     'filtType','bandpass', 'order',4,...
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

% define hyperparameters:
lambdas = [0.8 0.9 0.95 0.99...
           0.993 0.996 0.999]; % forgetting factor

alpha = 0.95;                  %  track a-prior error for misalignment
beta = 0.995;                  %  track noise floor for misalignment

m = 80;             

tic;
rls = optRLS(Mpreproc, binaryEvent1, m, lambdas, alpha, beta);
toc;


%% compare NMSE for each lambda for a representative voxel
[linear_idx] = sub2ind([dim.y dim.x], idx(2,1), idx(2,2));
labels = cell(1,length(lambdas));

% NMSE (output prediction) for S1
rls_nmse = rls.recursiveNMSE(:,linear_idx);
figure; 
sgtitle('Output Prediction Error','fontsize', 20, 'fontweight', 'b');
for i = 1:length(lambdas)
    plot(rls_nmse{i}, 'LineWidth', 2);
    hold on;
    labels{i} = sprintf('λ = %.4f', lambdas(i));
end
xlim([m dim.t]);
xlabel('Seconds');
ylabel('NMSE');
set(gca,'fontsize',18);
legend(labels);

% save 
saveas(gcf, fullfile(data_path, 's1_representative_nmse.fig'));

% Alignment for S1
se = rls.recursiveEMA_SE(:,linear_idx);
noise = rls.recursiveEMA_noise(:,linear_idx);

figure;
sgtitle('Alignment','fontsize', 20, 'fontweight', 'b');
for i = 1:length(lambdas)
    alignment = se{i} - noise{i};
    plot(alignment, 'LineWidth', 2);
    hold on;
    labels{i} = sprintf('λ = %.4f', lambdas(i));
end
xlim([m dim.t]);
xlabel('Seconds');
ylabel('e(t)^2 - \sigma_\eta^2');
set(gca,'fontsize',18);
legend(labels);


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

%% plot fit for representative voxel (all lambdas)
M2D = double(reshape(Mpreproc, numVoxels, dim.t));
linear_idx = sub2ind([dim.y dim.x], idx(2,1), idx(2,2));
y_original = M2D(linear_idx,:);
y = y_original(m:end); % truncate to match yhat length

% Loop through each lambda to generate and save figures
for i = 1:length(lambdas)
    lambda = lambdas(i);
    yhat = rls.all_y{i, linear_idx}';
    khat = rls.all_k{i, linear_idx};
    residuals = y - yhat;

    % Observed vs Predicted (overlay)
    figure;
    plot(y, 'k', 'LineWidth', 1); hold on;
    plot(yhat, 'm', 'LineWidth', 2);
    plot(khat(1,:), 'c', 'LineWidth', 2);
    legend('Observed','Predicted','Bias');
    title(sprintf('Observed vs. Predicted (Lambda = %.4f)', lambda));
    xlabel('Seconds'); ylabel('Amplitude');
    set(gca,'fontsize',16);
    xlim([0 length(yhat)]);
    hold off;
    saveas(gcf, fullfile(data_path, sprintf('S1_fit_lambda_%.4f.fig', lambda)));

    % Scatter plot yhat vs yobserved with linear fit
    figure;
    scatter(yhat, y, 'k'); hold on;
    p = polyfit(yhat, y, 1);
    yfit = polyval(p, yhat);
    plot(yhat, yfit, 'r', 'LineWidth', 2);
    title(sprintf('Observed vs. Predicted Scatter (Lambda = %.4f)', lambda));
    xlabel('Predicted yhat'); ylabel('Observed y');
    set(gca,'fontsize',16);
    axis square;
    hold off;
    saveas(gcf, fullfile(data_path, sprintf('S1_scatter_lambda_%.4f.fig', lambda)));

    % Residuals scatter plot
    figure;
    scatter(yhat, residuals, 'k');
    title(sprintf('Residuals vs. Predicted (Lambda = %.4f)', lambda));
    xlabel('Predicted'); ylabel('Residuals');
    axis square;
    set(gca,'fontsize',16);
    saveas(gcf, fullfile(data_path, sprintf('residuals_scatter_lambda_%.4f.fig', lambda)));

    % Time-Varying Bias Term
    figure;
    plot(khat(1,:), 'k', 'LineWidth', 2);
    title(sprintf('Bias Term (Lambda = %.4f)', lambda));
    xlabel('Seconds'); ylabel('Bias Amplitude');
    set(gca,'fontsize',16);
    xlim([0 length(khat(1,:))]);
    saveas(gcf, fullfile(data_path, sprintf('bias_term_lambda_%.4f.fig', lambda)));

    % Time-Varying Kernel (3D)
    figure;
    [kernel_len, time_len] = size(khat);
    [T, K] = meshgrid(1:time_len, 1:kernel_len);
    surf(T, K, khat, 'EdgeColor', 'interp');
    xlabel('Seconds'); ylabel('Kernel Coefficients');
    zlabel('Kernel Amplitude');
    title(sprintf('Time-Varying Kernel (Lambda = %.4f)', lambda));
    view([-30, 60]); shading interp;
    colormap parula; colorbar;
    set(gca,'fontsize',16);
    saveas(gcf, fullfile(data_path, sprintf('time_varying_kernel_lambda_%.4f.fig', lambda)));

    % Heatmap Kernel
    figure;
    imagesc(khat(2:end, :));
    cb = colorbar;
    colormap('parula');
    ylabel(cb, 'Kernel Amplitude (Z-Score)');
    xlabel('Seconds'); ylabel('Kernel Coefficients');
    set(gca,'fontsize',16);
    title(sprintf('Kernel Heatmap (Lambda = %.4f)', lambda));
    saveas(gcf, fullfile(data_path, sprintf('kernel_heatmap_lambda_%.4f.fig', lambda)));

    close all; % close figures to save memory
end
