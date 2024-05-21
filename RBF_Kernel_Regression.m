clear
clc
close all

% --- Simulation Parameters ---
carrierFreq = 3.5e9;             % Carrier frequency in Hz, the typical range for 5G NR (New Radio) systems.
pilotSpacing = 4;                % Spacing between pilot symbols
subcarrierSpacing = 15;          % Subcarrier spacing in kHz, common for LTE and 5G NR.
subcarriersPerRB = 12;           % 12 subcarriers per resource block, which is standard.
numResourceBlocks = 52;          % Number of resource blocks, a bandwidth with 52 resource blocks
totalSubcarriers = numResourceBlocks * subcarriersPerRB; % Total number of subcarriers
pilotIndices = (1:pilotSpacing:totalSubcarriers)'; % Indices of pilot symbols on the subcarrier, based on the pilot spacing.
numPilots = length(pilotIndices); % Number of pilot symbols

% --- Channel Parameters ---
numPaths = 20;                   % Number of multipath components, a multipath channel with 20 paths.
avgDelaySpread = 200e-9;         % Average delay spread in 200ns, typical for urban environments.

% --- Kernel Width Parameters for Testing ---
sigmaValues = linspace(1, 20, 20)'; % Test values for kernel width (sigma)
numTests = length(sigmaValues);     % Number of test values

% --- Simulation Setup ---
numTrials = 100;                  % Number of trials for each kernel width
signalToNoiseRatio = 10;          % Signal-to-noise ratio in dB

% --- Generate Random QPSK Symbols ---
order = 4;                        % QPSK=2^2
numBits=numPilots*log2(order);
bits = randi([0, 1], numBits, 1);
symbols = qammod(bits, order, 'InputType', 'bit', 'UnitAveragePower', true);

% --- Transmit Vector Initialization ---
txVector = zeros(totalSubcarriers, 1);
txVector(pilotIndices) = symbols;

% --- Initialize MSE Storage ---
mseValues = zeros(numTrials, numTests);
averageMSE = zeros(numTests, 1);

% --- Kernel Regression for Channel Estimation ---
for testIndex = 1:numTests
    currentSigma = sigmaValues(testIndex);

    for trialIndex = 1:numTrials
        % Generate random multipath channel
        channelResponse = generateRandomChannel(totalSubcarriers, avgDelaySpread, subcarrierSpacing);
        
        % Add noise to the received signal
        noiseVariance = db2pow(-signalToNoiseRatio);
        receivedSignal = channelResponse .* txVector + sqrt(noiseVariance / 2) * (randn(totalSubcarriers, 1) + 1i * randn(totalSubcarriers, 1));
        
        % Obtain raw channel estimates at pilot positions
        rawEstimates = receivedSignal(pilotIndices) ./ symbols;
        
        % Perform kernel regression to estimate the channel
        estimatedChannel = performRBFKernelRegression(pilotIndices, rawEstimates, totalSubcarriers, currentSigma);
        
        % Compute mean squared error
        mseValues(trialIndex, testIndex) = mean(abs(estimatedChannel - channelResponse).^2);
    end
    
    averageMSE(testIndex) = 10 * log10(mean(mseValues(:, testIndex)));
end

% --- Plot MSE vs Kernel Width ---
clf;
plot(sigmaValues, averageMSE, 'o-', 'LineWidth', 2);
grid on;
xlabel('Kernel Width (Sigma)', 'FontSize', 14);
ylabel('Mean Squared Error (dB)', 'FontSize', 14);
title('MSE vs Kernel Width');

% --- Find Optimal Kernel Width ---
[minimumMSE, optimalIndex] = min(averageMSE);
optimalSigma = sigmaValues(optimalIndex);

% --- Display Optimal Kernel Width and Corresponding MSE ---
fprintf('Optimal Kernel Width (Sigma):%7.2f with MSE: %7.2f dB\n', optimalSigma, minimumMSE);

% --- Simulate Transmission and Reception of OFDM Symbols ---
numOFDMSymbols = 50;                
transmittedMatrix = zeros(totalSubcarriers, numOFDMSymbols);
receivedMatrix = zeros(totalSubcarriers, numOFDMSymbols);
channelEstimatesMatrix = zeros(totalSubcarriers, numOFDMSymbols); % For storing channel estimates per symbol

for symbolIndex = 1:numOFDMSymbols
    % Generate and insert pilot symbols into TX vector for the symbol
    bits = randi([0, 1], numBits, 1);
    symbols = qammod(bits, order, 'InputType', 'bit', 'UnitAveragePower', true);
    
    % Pre-allocate a column vector for this OFDM symbol
    txVector = zeros(totalSubcarriers, 1);
    
    % Assign pilot symbols to their indices within the transmit vector
    txVector(pilotIndices) = symbols;

    % Store the transmitted data for this symbol
    transmittedMatrix(:, symbolIndex) = txVector;

    % Simulate the channel and noise for this symbol
    channelResponse = generateRandomChannel(totalSubcarriers, avgDelaySpread, subcarrierSpacing);
    noiseVariance = db2pow(-signalToNoiseRatio);
    receivedSignal = channelResponse .* txVector + sqrt(noiseVariance / 2) * (randn(totalSubcarriers, 1) + 1i * randn(totalSubcarriers, 1));
    
    % Store the channel-modulated data (for visualization of true channel conditions)
    transmittedMatrix(:, symbolIndex) = channelResponse;

    % Compute and store raw channel estimates at pilot positions
    rawChannelEstimates = receivedSignal(pilotIndices) ./ symbols;
    receivedMatrix(pilotIndices, symbolIndex) = rawChannelEstimates;    

    % Set kernel regression parameters for channel estimation
    kernelWidth = optimalSigma;
    
    % Perform kernel regression to estimate the channel
    estimatedChannel = performRBFKernelRegression(pilotIndices, rawChannelEstimates, totalSubcarriers, kernelWidth);
    
    % Store the kernel regression-based channel estimates
    channelEstimatesMatrix(:, symbolIndex) = estimatedChannel;
end

% Plotting the true, raw, and estimated channel conditions
figure;
surf(1:totalSubcarriers, 1:numOFDMSymbols, real(transmittedMatrix)', 'EdgeColor', 'none');
xlabel('Subcarrier Index');
ylabel('OFDM Symbol Index');
zlabel('|h[n,k]|');
title('True Channel Conditions');
colorbar;

% Kernel Regression Channel Estimates Plot
figure;
surf(1:totalSubcarriers, 1:numOFDMSymbols, real(channelEstimatesMatrix)', 'EdgeColor', 'none');
xlabel('Subcarrier Index');
ylabel('OFDM Symbol Index');
zlabel('|h[n,k]|');
title('RBF Kernel Regression Channel Estimates');
colorbar;

symbolsToPlot = 10;

% Loop through the first 10 symbols and create separate figures for each
for symbolIndex = 1:symbolsToPlot
    figure;

    % Extract the channel estimate for this symbol from the matrix
    estimatedChannel = channelEstimatesMatrix(:, symbolIndex);
    
    % Plot the true channel conditions
    plot(1:totalSubcarriers, real(transmittedMatrix(:, symbolIndex)), 'LineWidth', 2);
    hold on; % Hold on to plot the estimates on the same figure
    
    % Kernel Regression estimate across all subcarriers
    plot(1:totalSubcarriers, real(estimatedChannel), 'LineWidth', 2);
    
    hold off; 
    grid on; 
    xlabel('Subcarriers Index', 'FontSize', 14);
    ylabel(sprintf('|h[n,%d]|', symbolIndex), 'FontSize', 14);
    legend('True Channel', 'RBF Kernel Regression Estimate', 'Location', 'Best');
    xlim([0, totalSubcarriers]);
    title(sprintf('Channel Estimates for Symbol %d', symbolIndex));
end


function channelResponse = generateRandomChannel(numSubcarriers, avgDelaySpread, subcarrierSpacing)
    numPaths = 20;       % Number of multipath components
    delays = exprnd(avgDelaySpread, numPaths, 1);
    frequencies = subcarrierSpacing*1e3*(0:numSubcarriers-1)';
    initialPhases = unifrnd(0, 2*pi, 1, numPaths);
    phases = 2*pi*frequencies*delays' + initialPhases;
    channelResponse = sum(exp(1i*phases), 2) / sqrt(numPaths);
end


function estimatedChannel = performRBFKernelRegression(pilotIndices, receivedPilots, numSubcarriers, sigma)
    % Initialize vectors for the channel estimates and indicators.
    channelEstimates = zeros(numSubcarriers, 1);
    indicatorVector = zeros(numSubcarriers, 1);
    % Compute the length of the RBF kernel.
    kernelLength = floor(numSubcarriers / 10);
    kernelWeights = exp(-0.5 * (-kernelLength:kernelLength).^2 / sigma^2)';
    % Set the received pilots and indicators at the pilot indices.
    channelEstimates(pilotIndices) = receivedPilots;
    indicatorVector(pilotIndices) = 1; % Marking the presence of pilots.
    % Apply the RBF kernel to both channel estimates and indicators.
    [filteredEstimates, filteredEstimatesTransient] = filter(kernelWeights, 1, channelEstimates);
    filteredEstimates = [filteredEstimates(kernelLength+1:end); filteredEstimatesTransient(1:kernelLength)];
    [filteredIndicators, filteredIndicatorsTransient] = filter(kernelWeights, 1, indicatorVector);
    filteredIndicators = [filteredIndicators(kernelLength+1:end); filteredIndicatorsTransient(1:kernelLength)];
    % Compute the channel estimates.
    estimatedChannel = filteredEstimates ./ max(1e-8, filteredIndicators);
end