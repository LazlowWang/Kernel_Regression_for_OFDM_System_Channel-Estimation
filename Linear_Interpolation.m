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

% --- Simulation Setup ---
signalToNoiseRatio = 10;          

% --- Generate Random QPSK Symbols ---
order = 4;                      % QPSK=2^2
numBits=numPilots*log2(order);
bits = randi([0, 1], numBits, 1);
symbols = qammod(bits, order, 'InputType', 'bit', 'UnitAveragePower', true);

% --- Transmit Vector Initialization ---
txVector = zeros(totalSubcarriers, 1);
txVector(pilotIndices) = symbols;

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

    
    % Linear interpolation to estimate the channel for all subcarriers
    estimatedChannel =  interp1(pilotIndices, rawChannelEstimates, 1:totalSubcarriers, 'linear');
    
    % Store the Linear interpolation channel estimates
    channelEstimatesMatrix(:, symbolIndex) = estimatedChannel;
end

% Plotting the true, raw, and estimated channel conditions
% True Channel Conditions Plot
figure;
surf(1:totalSubcarriers, 1:numOFDMSymbols, real(transmittedMatrix)', 'EdgeColor', 'none');
xlabel('Subcarriers Index');
ylabel('OFDM Symbol Index');
zlabel('|h[n,k]|');
title('True Channel Conditions');
colorbar;

% Linear Interpolation Channel Estimates Plot
figure;
surf(1:totalSubcarriers, 1:numOFDMSymbols, real(channelEstimatesMatrix)', 'EdgeColor', 'none');
xlabel('Subcarriers Index');
ylabel('OFDM Symbol Index');
zlabel('|h[n,k]|');
title('Linear Interpolaion Channel Estimates');
colorbar;

symbolsToPlot = 10;

% Loop through the first 10 symbols and create separate figures for each
for symbolIndex = 1:symbolsToPlot
    figure; 

    % Extract the channel estimate for this symbol from the matrix
    estimatedChannel = channelEstimatesMatrix(:, symbolIndex);
    
    % Plot the true channel conditions
    plot(1:totalSubcarriers, real(transmittedMatrix(:, symbolIndex)), 'LineWidth', 2);
    hold on; 
    
    % Raw estimate 
    rawEstimates = receivedMatrix(:, symbolIndex); 
    plot(pilotIndices, real(rawEstimates(pilotIndices)), 'o', 'MarkerSize', 5); 
    
    plot(1:totalSubcarriers, real(estimatedChannel), 'LineWidth', 2); % Interpolated estimates
    
    hold off; 
    grid on; 
    xlabel('Subcarriers Index', 'FontSize', 14);
    ylabel(sprintf('|h[n,%d]|', symbolIndex), 'FontSize', 14);
    legend('True Channel', 'Raw Estimate', 'Linear Interpolation Estimate', 'Location', 'Best');
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
