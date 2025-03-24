function [rls] = optRLS(Y, u, m, lambdaCandidates, alpha, beta)
% optRLS - Optimize the forgetting factor (lambda) for a representative voxel using RLS.
%
%   INPUTS:
%       Y                - Observed output (3D: ydim x xdim x tdim)
%       u                - Input (1D vector)
%       m                - Number of impulse response coefficients
%       lambdaCandidates - Vector of candidate forgetting factor values (in [0, 1])
%       alpha            - forgetting factor for a-prior error (EMSE)
%       beta             - forgetting factor for floor noise (EMSE)
%
%   OUTPUTS:
%       rls - Struct containing MSE, NMSE, R², kernel estimates, and design matrix.

    %% Set up RLS
    [ydim, xdim, n] = size(Y);  
    numVoxels = ydim * xdim; 
    Y = reshape(Y, numVoxels, n);
    Y = Y'; 
                                  

    % Design matrix
    A = zeros(n, m);                           
    for i = 1:m
        A(i:end, i) = u(1:end - i + 1);
    end                                 
    A = [ones(n, 1), A]; % bias term  
    m = size(A,2);       % update the number of coefficients

    % Initialize variables for grid search over lambda 
    numLambda  = length(lambdaCandidates);
    r2   = cell(numLambda, numVoxels);            
    recursiveMSE  = cell(numLambda,numVoxels);             
    recursiveNMSE = cell(numLambda,numVoxels);
    recursiveEMA_SE = cell(numLambda,numVoxels);
    recursiveEMA_noise = cell(numLambda,numVoxels);
    all_k = cell(numLambda, numVoxels);                 
    all_yhat = cell(numLambda, numVoxels);              

    %% RLS for each candidate lambda 
    eps = 1e-3;     % stabilize NMSE calculation
    sigma = 1e3;    % initialization factor for the inverse covariance matrix
    parfor v = 1:numVoxels
        y = Y(:,v);
        for ii = 1:numLambda
            lambda = lambdaCandidates(ii);  
        
            % Initialize RLS parameters
            k = zeros(m,1);
            k_t = zeros(m,n);
            P = sigma * eye(m); %L2 norm

            % Initialize running values for PCV 
            runningMeanY = mean(y(1:m-1));
            runningVarY = var(y(1:m-1));
            running_mse = nan(1,n);                 % output prediction
            running_nmse = nan(1,n);                % output prediction
            ema_se = nan(1,n);
            ema_noise = nan(1,n);
            ema_se(m-1) = var(y(1:m-1)); 
            ema_noise(m-1) = var(y(1:m-1)); 

            % Recursion
            for t = (m-1):n  %truncate
                x = A(t,:)';

                % Kalman gain
                Px = P * x; 
                denom = lambda + (x' * Px);
                G = Px / denom;

                % A priori error
                yhat = x' * k;
                err = y(t) - yhat;

                % Update metrics
                if t > (m-1)
                    runningMeanY = ((t - 1) * runningMeanY + y(t)) / t;
                    deltaY = y(t) - runningMeanY;
                    runningVarY = ((t - 1) * runningVarY + deltaY^2) / t;
                    ema_se(t) = (alpha * ema_se(t-1)) + (1 - alpha) * err^2;
                    if norm(x)^2 == 0
                        ema_noise(t) = (beta * ema_noise(t-1)) + (1 - beta) * err^2;
                    else
                        ema_noise(t) = ema_noise(t-1);
                    end
                end

                % Update model evaluation metrics
                if t == (m-1)
                    running_mse(t) = err^2;
                    running_nmse(t) = running_mse(t) / (runningVarY + eps);
                else
                    numPoints = t - (m - 1) + 1;
                    running_mse(t) = (((numPoints-1) * running_mse(t - 1) + err^2) / numPoints);
                    running_nmse(t) = running_mse(t) / (runningVarY + eps);
                end

                % Update kernel
                k = k + G * err;

                % Save kernel estimate
                k_t(:,t) = k;

                % Update covariance
                P = (1/lambda) * (P - G * x' * P);
            end

            % calculate batch r^2
            batch_y = y(m-1:end);
            batch_yhat = sum(A(m-1:end,:) .* k_t(:,m-1:end)', 2);
            SS_res = sum((batch_y - batch_yhat).^2);
            SS_tot = sum((batch_y-mean(batch_y)).^2);
            r2{ii,v} = 1 - (SS_res / SS_tot);

            % Store lambda-specific parameters for each voxel
            all_k{ii,v} = k_t;
            all_yhat{ii,v} = batch_yhat;
            recursiveMSE{ii,v} = running_mse; 
            recursiveNMSE{ii,v} = running_nmse;
            recursiveEMA_SE{ii,v} = ema_se;
            recursiveEMA_noise{ii,v} = ema_noise;
        end
    end

    % save outputs
    rls = struct;
    rls.recursiveMSE = recursiveMSE;
    rls.recursiveNMSE = recursiveNMSE; 
    rls.recursiveEMA_SE = recursiveEMA_SE;
    rls.recursiveEMA_noise = recursiveEMA_noise;
    rls.r2 = r2; 
    rls.all_k = all_k;
    rls.all_y = all_yhat;
    rls.designMat = A;

end
