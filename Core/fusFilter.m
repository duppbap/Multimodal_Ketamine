function [Mfilt] = fusFilter(M,filt_params,dim)
%lpFilter Summary of this function goes here
% INPUT
% 1. M - fUSI data
% 2. Fs - sampling rate
% 3. Fc - cut off freq
%%


% Create main filter (bandpass, high-pass, or low-pass)
if strcmp(filt_params.filtType, 'bandpass')
    wn = [filt_params.fc1 filt_params.fc2] / (filt_params.fs / 2);
elseif strcmp(filt_params.filtType, 'high')
    wn = filt_params.fc1 / (filt_params.fs / 2);
elseif strcmp(filt_params.filtType, 'low')
    wn = filt_params.fc2 / (filt_params.fs / 2);
end
[b, a] = butter(filt_params.order, wn, filt_params.filtType);

% % Design Notch Filter using Butterworth Bandstop Filter
% notch_freq = 0.02 / (filt_params.fs / 2); % Normalize to Nyquist frequency
% notch_bw = 0.01 / (filt_params.fs / 2);  % Define narrow bandwidth (adjustable)
% notch_wn = [notch_freq - notch_bw, notch_freq + notch_bw]; % Bandstop range
% [bn, an] = butter(2, notch_wn, 'stop'); % 2nd-order Butterworth bandstop filter

% Reshape to 2D if 3D
if ndims(M) == 3
    M = reshape(M, dim.y * dim.x, []);
end

% Apply Filters
numVox = size(M, 1);
tdim = size(M, 2);
Mfilt = zeros(numVox, tdim);

for v = 1:numVox
    x = M(v, :);
    x = filtfilt(b, a, x);   % Apply main filter
    % x = filtfilt(bn, an, x); % Apply Butterworth notch (bandstop) filter
    Mfilt(v, :) = x;
end

% Reshape back to 3D
Mfilt = reshape(Mfilt, dim.y, dim.x, []);



end