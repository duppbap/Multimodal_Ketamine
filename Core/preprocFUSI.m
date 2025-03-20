function [Mpreproc] = preprocFUSI(mat3D,filt_params,norm_params)
%preprocFUSI preprocesses fUSI data. 
% motion correction, removal of noisy segment, and baseline normalization
% (no ephys)
% if filtOpt is true, then highpass filter before baseline normalization 
% INPUTS
% 1. filename
% 2. noise - start and end of where ephys is noisy
% 3. filtOpt - boolean for filter or no filter.
% OUTPUTS
% 1. Mpreproc - unfiltered
%%
%piecewise motion correction
showVid = 0;
M = MotionCorr(mat3D, showVid);



% z score normalization
bslMED = mean(M(:,:,1:norm_params.duration), 3);
bslSTD = std(M(:,:,1:norm_params.duration),[], 3);
M = (M - bslMED) ./ (bslSTD + 1e-3);

%filter
if filt_params.filtOpt == 1
    [dim.y, dim.x, dim.t] = size(M);
    M = fusFilter(M,filt_params,dim);
end

% save output
Mpreproc = M;

%%
end