function [hrf, A, R2, MSE] = computeIR(Y,u,m,lambda)
% estimate impulse response for a LTI system where the output 
% is expressed as a convolution of the input with the impulse response
% AUTHOR: Wooseong Choi
% LAST UPDATE: 20241115
% INPUT
%   u - input (binary event) vector (1D) 
%   m - number of coefficients in the impulse response
%   Y - observed output matrix (3D - ydim x xdim x tdim)
%   lambda - l2 regularization term
%   pps - event time stamps
%   drift_start - start of the drift term
%
% OUTPUT
%   hrfOUT - impulse response (2D - numVox x n)
%   R2OUT - variance explained
%   A - design matrix
%
% NOTES
% Define A as regression matrix where each row contains past input values
% First column of A is for the bias term
% Second column of A is for the quadratic drift
%%

% Check that time points for input and outputs match
if length(u) ~= size(Y, 3)
    disp('Error: Length of Input and Output Do Not Match');
    return;
end

% reshape
[ydim, xdim, tdim] = size(Y);
numVox = ydim * xdim;
Y = reshape(Y, numVox, tdim);

% construct design matrix
n = size(Y,2);
X = zeros(n, m);
for i = 1:m
    X(i:end,i) = u(1:end-i+1);
end

%column for the intercept
int = ones(n,1);

%horizontal concantenation
A = [int X];

%update number of coeff
numCoeff = size(A,2);

%regularized A
P = eye(size(A' * A)); %m-by-m
P(:,1) = 0;          % do not regularize the bias and drift terms
Areg = (A' * A) + (lambda * P); 


%%  estimate coefficients:

%initialize variables
x = zeros(numCoeff, numVox);  %kernel coefficients
adjR2 = zeros(numVox,1);      % R^2 values
MSE = zeros(numVox,1);

for v = 1:numVox
    %collect voxel data
    y = Y(v, :)';

    %L2 regularized least squares
    x(:, v) = Areg \ (A' * y);

    %calculate R squared
    y_avg = mean(y);
    yhat = A * x(:,v);
    SS_reg = sum((yhat - y_avg).^2, 'all');
    SS_res = sum((y - yhat).^2, 'all');
    MSE(v) = ((1/tdim) * SS_res) / var(y,0,1);
    SS_tot = SS_reg + SS_res;
    adjR2(v) = (SS_reg / SS_tot);
end
disp('Finished computing HRF for each voxel');


%reshape outputs
hrf = reshape(x, numCoeff, ydim, xdim);
R2 = reshape(adjR2, ydim, xdim, 1);



%% Plot
font_size = 12; 
[~, max_idx] = max(R2(:));  % Find index of max R²
[row, col] = ind2sub([ydim, xdim], max_idx);  % Convert to row, col coordinates

%collect kernel and data for voxel with max R2.
k = squeeze(hrf(:,row,col));
yMaxHat = A * k;
ii = sub2ind([ydim xdim tdim], row, col);
yMax = Y(ii,:)';

%plot example
ts = 1:length(yMaxHat);
figure; 
set(gcf, 'windowstyle', 'docked');
plot(ts,yMaxHat, 'r', 'LineWidth', 2);
hold on;
plot(ts,yMax,'b','LineWidth', 1);
ylim([-100 400]);

% for i = 1:size(pps,1)
%     t1 = (pps(i,1));
%     t2 = (pps(i,2));
%     y_limits = ylim;
%     patch([t1 t2 t2 t1],...
%         [y_limits(1) y_limits(1) y_limits(2) y_limits(2)],...
%         'k', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
% end
title('Model Fit');
xlabel('Seconds', 'FontSize', font_size);
xlim([0 (length(ts))]);
legend('Predicted','Observed');
ax = gca;
ax.FontSize = font_size;

%plot kernel
figure; set(gcf,'windowstyle','docked');
%kts = ((0:m-1))';
kernel = k(3:end);
time_vector = 0:1:length(kernel)-1;
plot(time_vector, kernel,'k','LineWidth',3); %dont plot bias and drift terms
xlabel('Seconds');
ylabel('Amplitude');
xlim([0 m-2])
title('Kernel');
set(gca,'fontsize', font_size);
axis square;


%% scatter (observed vs predicted Y)
figure; 
set(gcf, 'windowstyle', 'docked');
rr = R2(row,col);
set(gcf, 'windowstyle', 'docked');
scatter(yMaxHat, yMax, 10, 'k','LineWidth', 1);
hold on ;
plotLine = -100:600;
maxR2vector = plotLine * rr;
plot(plotLine,maxR2vector,'LineWidth',3,'Color','r');
xlim([-100 600]);
xlabel('Predicted');
ylabel('Observed');
axis square;
ax = gca;
ax.FontSize = font_size;

% scatter without the baseline terms
% figure; 
% set(gcf, 'windowstyle', 'docked');
% drift = A(:,2) * k(2,:);
% res1 = yMax - drift; 
% res2 = yMaxHat - drift;
% scatter(res2, res1, 'k','LineWidth',1);
% axis square;
% title('Predicted vs Observed (w/o Drift Terms)')
% xlabel('Predicted');
% ylabel('Observed');
% ax = gca;
% ax.FontSize = font_size;

% residual
figure; 
set(gcf, 'windowstyle', 'docked');
residual = yMax - yMaxHat; 
residual = residual ./ std(residual);
scatter(yMaxHat, residual, 'k', 'LineWidth', 1);
xlabel('Predicted');
ylabel('Standardized Residual');
xlim([0 200]);
axis square;
ax = gca; 
ax.FontSize = font_size;

%% Explained variance specific to PPS

% % calculate portion of output without drift
% hrf_2D = reshape(hrf, size(hrf,1), dim.y * dim.x);
% y_drift = (A(:,2) * hrf_2D(2,:))';
% y = reshape(Mpreproc, dim.y * dim.x , []);
% y_delta = y - y_drift;
% 
% % calculate output only predicted by PPS
% A_pps = [A(:,1) A(:,3:end)];
% hrf_pps = [hrf_2D(1,:); hrf_2D(3:end,:)];
% y_pps = (A_pps * hrf_pps)';
% 
% % calculate variance explained
% SS1 = sum((y_pps - y_delta).^2,2);
% y_mean = mean(y_delta,2);
% SS2 = sum((y_delta - y_mean).^2,2 );
% R2_pps = 1 - (SS1./SS2);
% R2_map = reshape(R2_pps, dim.y, dim.x);
% 
% figure; 
% imagesc(R2_map);
% clim([0 1]);



%% Explained variance map 

% figure; 
% set(gcf, 'windowstyle', 'docked');
% imagesc(R2);
% colormap('parula');
% %clim([0 1]);
% cb = colorbar;
% sgtitle('Explained Variance Map');
% ylabel(cb, 'R squared');

%
end