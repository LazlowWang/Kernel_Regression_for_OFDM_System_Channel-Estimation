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
sigmaValues = linspace(1, 25, 10)'; % Test values for kernel width (sigma)
numTests = length(sigmaValues);     % Number of test values

% SNR Range for BER calculation
SNRRange = 0:5:30;
BERTrueChannel = zeros(length(SNRRange), 1);
BEREstimatedChannel = zeros(length(SNRRange), 1);
KBEREstimatedChannel = zeros(length(SNRRange), 1);
LBEREstimatedChannel = zeros(length(SNRRange), 1); 

% --- Simulation Setup ---
numTrials = 100;                  % Number of trials for each kernel width
signalToNoiseRatio = 10;          % Signal-to-noise ratio in dB

% --- Generate Random QPSK Symbols ---
bitsPerPilot = 2;
modulationOrder = 2^bitsPerPilot;
pilotBits = randi([0, 1], numPilots * bitsPerPilot, 1);
pilotSymbols = qammod(pilotBits, modulationOrder, 'UnitAveragePower', true, 'InputType', 'bit');

% --- Transmit Vector Initialization ---
txVector = zeros(totalSubcarriers, 1);
txVector(pilotIndices) = pilotSymbols;

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
        rawEstimates = receivedSignal(pilotIndices) ./ pilotSymbols;
        
        % Perform kernel regression to estimate the channel
        estimatedChannel = performRBFKernelRegression(pilotIndices, rawEstimates, totalSubcarriers, currentSigma);
        
        % Compute mean squared error
        mseValues(trialIndex, testIndex) = mean(abs(estimatedChannel - channelResponse).^2);
    end
    
    averageMSE(testIndex) = 10 * log10(mean(mseValues(:, testIndex)));
end

% --- Find Optimal Kernel Width ---
[minimumMSE, optimalIndex] = min(averageMSE);
optimalSigma = sigmaValues(optimalIndex);

% --- Simulate Transmission and Reception of OFDM Symbols ---
numOFDMSymbols = 50;                % Number of OFDM symbols to process

% Loop over SNR values
for snrIndex = 1:length(SNRRange)
    SNR = SNRRange(snrIndex);
    bitErrorsTrue = 0;
    bitErrorsEstimated = 0;
    totalBits = 0;

    for symbolIndex = 1:numOFDMSymbols
        % Generate random bits and corresponding QAM symbols for the entire frame
        dataBits = randi([0, 1], (totalSubcarriers - numPilots) * bitsPerPilot, 1);
        dataSymbols = qammod(dataBits, modulationOrder, 'UnitAveragePower', true, 'InputType', 'bit');
        
        % Create a new TX vector that includes both pilots and data
        txVector = zeros(totalSubcarriers, 1);
        txVector(pilotIndices) = pilotSymbols;
        dataIndices = setdiff(1:totalSubcarriers, pilotIndices)';
        txVector(dataIndices) = dataSymbols;
        
        % Simulate the channel
        channelResponse = generateRandomChannel(totalSubcarriers, avgDelaySpread, subcarrierSpacing);
        transmittedMatrix(:, symbolIndex) = channelResponse;

        % Add noise
        noiseVariance = db2pow(-SNR);
        receivedSignal = channelResponse .* txVector + sqrt(noiseVariance / 2) * (randn(totalSubcarriers, 1) + 1i * randn(totalSubcarriers, 1));
        receivedMatrix(:, symbolIndex) = receivedSignal;

        % Channel estimation at pilot positions
        rawChannelEstimates = receivedSignal(pilotIndices) ./ pilotSymbols;

        % Kernel regression for channel estimation
        estimatedChannel = performRBFKernelRegression(pilotIndices, rawChannelEstimates, totalSubcarriers, optimalSigma);
        KchannelEstimatesMatrix(:, symbolIndex) = estimatedChannel;

        % Equalize the received signal using the true and estimated channels
        equalizedSignalTrue = receivedSignal ./ channelResponse;
        equalizedSignalEstimated = receivedSignal ./ estimatedChannel;

        % Demodulate
        receivedBitsTrue = qamdemod(equalizedSignalTrue(dataIndices), modulationOrder, 'OutputType', 'bit', 'UnitAveragePower', true);
        receivedBitsEstimated = qamdemod(equalizedSignalEstimated(dataIndices), modulationOrder, 'OutputType', 'bit', 'UnitAveragePower', true);

        % Calculate bit errors
        bitErrorsTrue = bitErrorsTrue + sum(dataBits ~= receivedBitsTrue);
        bitErrorsEstimated = bitErrorsEstimated + sum(dataBits ~= receivedBitsEstimated);
        totalBits = totalBits + length(dataBits);

     end

        % Calculate and store BER for this SNR
        KBEREstimatedChannel(snrIndex) = bitErrorsEstimated / totalBits;
end


% Loop over SNR values
for snrIndex = 1:length(SNRRange)
    SNR = SNRRange(snrIndex);
    bitErrorsTrue = 0;
    bitErrorsEstimated = 0;
    totalBits = 0;

    for symbolIndex = 1:numOFDMSymbols
        % Generate random bits and corresponding QAM symbols for the entire frame
        dataBits = randi([0, 1], (totalSubcarriers - numPilots) * bitsPerPilot, 1);
        dataSymbols = qammod(dataBits, modulationOrder, 'UnitAveragePower', true, 'InputType', 'bit');
        
        % Create a new TX vector that includes both pilots and data
        txVector = zeros(totalSubcarriers, 1);
        txVector(pilotIndices) = pilotSymbols;
        dataIndices = setdiff(1:totalSubcarriers, pilotIndices)';
        txVector(dataIndices) = dataSymbols;
        
        % Simulate the channel
        channelResponse = generateRandomChannel(totalSubcarriers, avgDelaySpread, subcarrierSpacing);
        transmittedMatrix(:, symbolIndex) = channelResponse;
        
        % Add noise
        noiseVariance = db2pow(-SNR);
        receivedSignal = channelResponse .* txVector + sqrt(noiseVariance / 2) * (randn(totalSubcarriers, 1) + 1i * randn(totalSubcarriers, 1));
        receivedMatrix(:, symbolIndex) = receivedSignal;

        % Channel estimation at pilot positions
        rawChannelEstimates = receivedSignal(pilotIndices) ./ pilotSymbols;

        % Kernel regression for channel estimation
        estimatedChannel = performGaussianKernelRegression(pilotIndices, rawChannelEstimates, totalSubcarriers, optimalSigma);
        channelEstimatesMatrix(:, symbolIndex) = estimatedChannel;

        % Equalize the received signal using the true and estimated channels
        equalizedSignalTrue = receivedSignal ./ channelResponse;
        equalizedSignalEstimated = receivedSignal ./ estimatedChannel;

        % Demodulate
        receivedBitsTrue = qamdemod(equalizedSignalTrue(dataIndices), modulationOrder, 'OutputType', 'bit', 'UnitAveragePower', true);
        receivedBitsEstimated = qamdemod(equalizedSignalEstimated(dataIndices), modulationOrder, 'OutputType', 'bit', 'UnitAveragePower', true);

        % Calculate bit errors
        bitErrorsTrue = bitErrorsTrue + sum(dataBits ~= receivedBitsTrue);
        bitErrorsEstimated = bitErrorsEstimated + sum(dataBits ~= receivedBitsEstimated);
        totalBits = totalBits + length(dataBits);

     end

        % Calculate and store BER for this SNR
        BERTrueChannel(snrIndex) = bitErrorsTrue / totalBits;
        RKBEREstimatedChannel(snrIndex) = bitErrorsEstimated / totalBits;
end

% Plot BER vs. SNR
figure;
semilogy(SNRRange, BERTrueChannel, 'b-o', SNRRange, KBEREstimatedChannel, 'b--x', SNRRange, RKBEREstimatedChannel, 'black--x');
xlabel('SNR (dB)');
ylabel('BER');
legend('Theoretical', 'RBF Kernel Regression Estimated', 'Gaussian Kernel Regression Estimated');
title('BER vs. SNR');
grid on;


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
    % Compute the unnormalized channel estimate.
    estimatedChannel = filteredEstimates ./ max(1e-8, filteredIndicators);
end


function estimatedChannel = performGaussianKernelRegression(pilotIndices, receivedPilots, numSubcarriers, sigma)
    % Create a grid for the subcarriers.
    subcarrierGrid = (1:numSubcarriers)';
    estimatedChannel = zeros(numSubcarriers, 1);
    
    % Loop through each subcarrier to compute its channel estimate.
    for i = 1:numSubcarriers
        % Calculate squared distances from current subcarrier to each pilot.
        distances = (subcarrierGrid(i) - pilotIndices).^2;
        % Calculate weights using the Gaussian kernel.
        weights = exp(-distances / (2 * sigma^2));
        % Compute the weighted sum of received pilots.
        weightedSum = sum(weights .* receivedPilots);
        % Sum of weights for normalization.
        sumWeights = sum(weights);
        % Normalize the weighted sum to estimate the channel at this subcarrier.
        estimatedChannel(i) = weightedSum / sumWeights;
    end
end
