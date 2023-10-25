clear;
clc;
close all;

%%
% Initial Parameters

add_awgn = true;                    % AWGN eklensin mi
add_iq_imbalance = true;            % IQ Imbalance Eklensin mi

iq_imbalance_compensation = true;   % IQ Dengesizliği düzeltmesi açık/kapalı (açık = true)
blind_algorithm = true;            % Blind Algoritmanın mı yoksa MATLAB algoritmasının mı çalışacağını belirler (true = Blind Algorithm)
window_size = 500;                  % Blind Algoritmada ki hareketli ortalamanın boyutunu (window size) belirler

amplitude_imblance = 2.5;           % genlik dengesizliği (dB)
phase_imbalance = 10;               % faz dengesizliği (derece)
snr_level = 20;                     % SNR seviyesi (dB)

display_spectrum_analyzer = false;      % spektrum analizörü göster
display_time_scope = false;             % time scope'u göster
display_constellation_diagram = true;  % constellation diagram'ı göster
display_tx_rx_image = true;            % iletilen ve alınan resmi göster

% IQ Imbalance Compensation Algorithms: line 145

%%

% TRANSMITTER
% -------------------------------------------------------------------------

%Configure all the scopes and figures for the example.

% Setup handle for image plot
if ~exist('imFig','var') || ~ishandle(imFig) %#ok<SUSENS> 
    imFig = figure;
    imFig.NumberTitle = 'off';
    imFig.Name = 'Image Plot';
    imFig.Visible = 'off';
else
    clf(imFig); % Clear figure
    imFig.Visible = 'off';
end

% Setup Spectrum viewer
spectrumScope = dsp.SpectrumAnalyzer( ...
'SpectrumType','Power density', ...
'SpectralAverages',10, ...
'YLimits',[-90 -30], ...
'Title','Received Baseband WLAN Signal Spectrum', ...
'YLabel','Power spectral density', ...
'Position',[69 376 800 450]);


% Setup the constellation diagram viewer for equalized WLAN symbols
refQAM = wlanReferenceSymbols('64QAM');
constellation = comm.ConstellationDiagram(...
    Title='Equalized WLAN Symbols',...
    ShowReferenceConstellation=true,...
    ReferenceConstellation=refQAM,...
    Position=[878 376 460 460]);

%%
% Prepare Image File

% Input an image file and convert to binary stream
fileTx = 'peppers.png';                          % Image file name
fData = imread(fileTx);                          % Read image data from file
scale = 0.3;                                     % Image scaling factor
origSize = size(fData);                          % Original input image size
scaledSize = max(floor(scale.*origSize(1:2)),1); % Calculate new image size
heightIx = min(round(((1:scaledSize(1))-0.5)./scale+0.5),origSize(1));
widthIx = min(round(((1:scaledSize(2))-0.5)./scale+0.5),origSize(2));
fData = fData(heightIx,widthIx,:);               % Resize image
imsize = size(fData);                            % Store new image size
txImage = fData(:);

if (display_tx_rx_image)
    % Plot transmit image
    imFig.Visible = 'on';
    subplot(211);
    imshow(fData);
    title('Transmitted Image');
    subplot(212);
    title('Received image appears here...');
    set(gca,'Visible','off');
end

set(findall(gca, 'type', 'text'), 'visible', 'on');

%%
%Fragment Transmit Data

msduLength = 2304; % MSDU length in bytes
numMSDUs = ceil(length(txImage)/msduLength);
padZeros = msduLength-mod(length(txImage),msduLength);
txData = [txImage;zeros(padZeros,1)];
txDataBits = double(reshape(de2bi(txData, 8)',[],1));

% Divide input data stream into fragments
bitsPerOctet = 8;
data = zeros(0,1);

for i=0:numMSDUs-1

    % Extract image data (in octets) for each MPDU
    frameBody = txData(i*msduLength+1:msduLength*(i+1),:);

    % Create MAC frame configuration object and configure sequence number
    cfgMAC = wlanMACFrameConfig(FrameType='Data',SequenceNumber=i);

    % Generate MPDU
    [psdu, lengthMPDU]= wlanMACFrame(frameBody,cfgMAC,OutputFormat='bits');

    % Concatenate PSDUs for waveform generation
    data = [data; psdu]; %#ok<AGROW>

end

%%
% Generate 802.11a Baseband WLAN Signal

nonHTcfg = wlanNonHTConfig;       % Create packet configuration
nonHTcfg.MCS = 6;                 % Modulation: 64QAM Rate: 2/3
nonHTcfg.NumTransmitAntennas = 1; % Number of transmit antenna
chanBW = nonHTcfg.ChannelBandwidth;
nonHTcfg.PSDULength = lengthMPDU; % Set the PSDU length

scramblerInitialization = randi([1 127],numMSDUs,1);

osf = 1.5;

sampleRate = wlanSampleRate(nonHTcfg); % Nominal sample rate in Hz

% Generate baseband NonHT packets separated by idle time
txWaveform = wlanWaveformGenerator(data,nonHTcfg, ...
'NumPackets',numMSDUs,'IdleTime',20e-6, ...
'ScramblerInitialization',scramblerInitialization, ...
'OversamplingFactor',osf);

%%
% RECEIVER

% Add AWGN and IQ Imbalance
if add_awgn == true && add_iq_imbalance == false
    rxWaveform = awgn(txWaveform,snr_level,'measured');                     % Sinyal awgn kanalından geçirilir
end

if add_awgn == false && add_iq_imbalance == true
    rxWaveform = iqimbal(txWaveform, amplitude_imblance,phase_imbalance);   % Girilen değerlerde IQ Imbalance Eklenir
end

if add_awgn == true && add_iq_imbalance == true
    rxWaveform_temp = awgn(txWaveform,snr_level,'measured');                     % Sinyal awgn kanalından geçirilir
    rxWaveform = iqimbal(rxWaveform_temp, amplitude_imblance,phase_imbalance);   % Girilen değerlerde IQ Imbalance Eklenir
end

if add_awgn == false && add_iq_imbalance == false
    rxWaveform = txWaveform;
end

% rxWaveform = awgn(txWaveform,snr_level,'measured');                     % Sinyal awgn kanalından geçirilir
% rxWaveform = iqimbal(rxWaveform, amplitude_imblance,phase_imbalance);   % Girilen değerlerde IQ Imbalance Eklenir

rxWaveform_imbalanced = rxWaveform;     %Time Scope için Compensation işleminden önceki sinyal değeri farklı bir değişkene kadyedilir

%%

% IQ Imbalance Compensation Algorithms Section
if (iq_imbalance_compensation)

    if (blind_algorithm == false) % MATLAB IQ Imbalance Compensation Algorithm

        iqComp = comm.IQImbalanceCompensator('CoefficientOutputPort',true);
        [compSig,coef] = iqComp(rxWaveform);
        rxWaveform = compSig;

    else % Blind IQ Imbalance Compensation Algorithm

        theta_1=-1.*movmean(sign(real(rxWaveform)).*imag(rxWaveform),[window_size-1 0]);
        theta_2=movmean(abs(real(rxWaveform)),[window_size-1 0]);
        theta_3=movmean(abs(imag(rxWaveform)),[window_size-1 0]);

        c1=theta_1./theta_2;
        c2=sqrt((theta_3.^2-theta_1.^2)./(theta_2.^2));

        rxWaveform_I = real(rxWaveform).*c2;
        rxWaveform_Q = real(rxWaveform).*c1 + imag(rxWaveform);
        rxWaveform = rxWaveform_I + 1i.*rxWaveform_Q;
    end
end


%%

%Spectrums

if (display_time_scope)
    if (iq_imbalance_compensation)
        scopes = timescope;
        scopes(txWaveform, rxWaveform_imbalanced, rxWaveform );
    else
        scopes = timescope;
        scopes(txWaveform, rxWaveform );
    end
end

if (display_spectrum_analyzer)
    spectrumScope.SampleRate = sampleRate*osf;
    spectrumScope(rxWaveform);
    release(spectrumScope);
end

%%

% Receiver Processing

aStop = 40;                                             % Stopband attenuation
ofdmInfo = wlanNonHTOFDMInfo('NonHT-Data',nonHTcfg);    % OFDM parameters
SCS = sampleRate/ofdmInfo.FFTLength;                    % Subcarrier spacing
txbw = max(abs(ofdmInfo.ActiveFrequencyIndices))*2*SCS; % Occupied bandwidth
[L,M] = rat(1/osf);
maxLM = max([L M]);
R = (sampleRate-txbw)/sampleRate;
TW = 2*R/maxLM;                                         % Transition width
b = designMultirateFIR(L,M,TW,aStop);

firrc = dsp.FIRRateConverter(L,M,b);
rxWaveform = firrc(rxWaveform);

displayFlag = false; 

rxWaveformLen = size(rxWaveform,1);
searchOffset = 0; % Offset from start of the waveform in samples

ind = wlanFieldIndices(nonHTcfg);
Ns = ind.LSIG(2)-ind.LSIG(1)+1; % Number of samples in an OFDM symbol

% Minimum packet length is 10 OFDM symbols
lstfLen = double(ind.LSTF(2)); % Number of samples in L-STF
minPktLen = lstfLen*5;
pktInd = 1;
fineTimingOffset = [];
packetSeq = [];
rxBit = [];

% Perform EVM calculation
evmCalculator = comm.EVM(AveragingDimensions=[1 2 3]);
evmCalculator.MaximumEVMOutputPort = true;

%%


while (searchOffset+minPktLen)<=rxWaveformLen
    % Packet detect
    pktOffset = wlanPacketDetect(rxWaveform,chanBW,searchOffset,0.5);

    % Adjust packet offset
    pktOffset = searchOffset+pktOffset;
    if isempty(pktOffset) || (pktOffset+double(ind.LSIG(2))>rxWaveformLen)
        if pktInd==1
            disp('** No packet detected **');
        end
        break;
    end

    % Extract non-HT fields and perform coarse frequency offset correction
    % to allow for reliable symbol timing
    nonHT = rxWaveform(pktOffset+(ind.LSTF(1):ind.LSIG(2)),:);
    coarseFreqOffset = wlanCoarseCFOEstimate(nonHT,chanBW);
    nonHT = frequencyOffset(nonHT,sampleRate,-coarseFreqOffset);

    % Symbol timing synchronization
    fineTimingOffset = wlanSymbolTimingEstimate(nonHT,chanBW);

    % Adjust packet offset
    pktOffset = pktOffset+fineTimingOffset;

    % Timing synchronization complete: Packet detected and synchronized
    % Extract the non-HT preamble field after synchronization and
    % perform frequency correction
    if (pktOffset<0) || ((pktOffset+minPktLen)>rxWaveformLen)
        searchOffset = pktOffset+1.5*lstfLen;
        continue;
    end
    fprintf('\nPacket-%d detected at index %d\n',pktInd,pktOffset+1);

    % Extract first 7 OFDM symbols worth of data for format detection and
    % L-SIG decoding
    nonHT = rxWaveform(pktOffset+(1:7*Ns),:);
    nonHT = frequencyOffset(nonHT,sampleRate,-coarseFreqOffset);

    % Perform fine frequency offset correction on the synchronized and
    % coarse corrected preamble fields
    lltf = nonHT(ind.LLTF(1):ind.LLTF(2),:);           % Extract L-LTF
    fineFreqOffset = wlanFineCFOEstimate(lltf,chanBW);
    nonHT = frequencyOffset(nonHT,sampleRate,-fineFreqOffset);
    cfoCorrection = coarseFreqOffset+fineFreqOffset; % Total CFO

    % Channel estimation using L-LTF
    lltf = nonHT(ind.LLTF(1):ind.LLTF(2),:);
    demodLLTF = wlanLLTFDemodulate(lltf,chanBW);
    chanEstLLTF = wlanLLTFChannelEstimate(demodLLTF,chanBW);

    % Noise estimation
    noiseVarNonHT = helperNoiseEstimate(demodLLTF);

    % Packet format detection using the 3 OFDM symbols immediately
    % following the L-LTF
    format = wlanFormatDetect(nonHT(ind.LLTF(2)+(1:3*Ns),:), ...
        chanEstLLTF,noiseVarNonHT,chanBW);
    disp(['  ' format ' format detected']);
    if ~strcmp(format,'Non-HT')
        fprintf('  A format other than Non-HT has been detected\n');
        searchOffset = pktOffset+1.5*lstfLen;
        continue;
    end

    % Recover L-SIG field bits
    [recLSIGBits,failCheck] = wlanLSIGRecover( ...
        nonHT(ind.LSIG(1):ind.LSIG(2),:), ...
        chanEstLLTF,noiseVarNonHT,chanBW);

    if failCheck
        fprintf('  L-SIG check fail \n');
        searchOffset = pktOffset+1.5*lstfLen;
        continue;
    else
        fprintf('  L-SIG check pass \n');
    end

    % Retrieve packet parameters based on decoded L-SIG
    [lsigMCS,lsigLen,rxSamples] = helperInterpretLSIG(recLSIGBits,sampleRate);

    if (rxSamples+pktOffset)>length(rxWaveform)
        disp('** Not enough samples to decode packet **');
        break;
    end

    % Apply CFO correction to the entire packet
    rxWaveform(pktOffset+(1:rxSamples),:) = frequencyOffset(...
        rxWaveform(pktOffset+(1:rxSamples),:),sampleRate,-cfoCorrection);

    % Create a receive Non-HT config object
    rxNonHTcfg = wlanNonHTConfig;
    rxNonHTcfg.MCS = lsigMCS;
    rxNonHTcfg.PSDULength = lsigLen;

    % Get the data field indices within a PPDU
    indNonHTData = wlanFieldIndices(rxNonHTcfg,'NonHT-Data');

    % Recover PSDU bits using transmitted packet parameters and channel
    % estimates from L-LTF
    [rxPSDU,eqSym] = wlanNonHTDataRecover(rxWaveform(pktOffset+...
        (indNonHTData(1):indNonHTData(2)),:), ...
        chanEstLLTF,noiseVarNonHT,rxNonHTcfg);

    % Show current constellation
    if (display_constellation_diagram)
        constellation(reshape(eqSym,[],1)); 
        release(constellation);
    end

    refSym = wlanClosestReferenceSymbol(eqSym,rxNonHTcfg);
    [evm.RMS,evm.Peak] = evmCalculator(refSym,eqSym);

    % Decode the MPDU and extract MSDU
    [cfgMACRx,msduList{pktInd},status] = wlanMPDUDecode(rxPSDU,rxNonHTcfg); %#ok<*SAGROW>

    if strcmp(status,'Success')
        disp('  MAC FCS check pass');

        % Store sequencing information
        packetSeq(pktInd) = cfgMACRx.SequenceNumber;

        % Convert MSDU to a binary data stream
        rxBit{pktInd} = reshape(de2bi(hex2dec(cell2mat(msduList{pktInd})),8)',[],1);

    else % Decoding failed
        if strcmp(status,'FCSFailed')
            % FCS failed
            disp('  MAC FCS check fail');
        else
            % FCS passed but encountered other decoding failures
            disp('  MAC FCS check pass');
        end

        % Since there are no retransmissions modeled in this example, we
        % extract the image data (MSDU) and sequence number from the MPDU,
        % even though FCS check fails.

        % Remove header and FCS. Extract the MSDU.
        macHeaderBitsLength = 24*bitsPerOctet;
        fcsBitsLength = 4*bitsPerOctet;
        msduList{pktInd} = rxPSDU(macHeaderBitsLength+1:end-fcsBitsLength);

        % Extract and store sequence number
        sequenceNumStartIndex = 23*bitsPerOctet+1;
        sequenceNumEndIndex = 25*bitsPerOctet-4;
        packetSeq(pktInd) = bi2de(rxPSDU(sequenceNumStartIndex:sequenceNumEndIndex)');

        % MSDU binary data stream
        rxBit{pktInd} = double(msduList{pktInd});
    end

    % Display decoded information
    if displayFlag
        fprintf('  Estimated CFO: %5.1f Hz\n\n',cfoCorrection); %#ok<*UNRCH> 

        disp('  Decoded L-SIG contents: ');
        fprintf('                            MCS: %d\n',lsigMCS);
        fprintf('                         Length: %d\n',lsigLen);
        fprintf('    Number of samples in packet: %d\n\n',rxSamples);

        fprintf('  EVM:\n');
        fprintf('    EVM peak: %0.3f%%  EVM RMS: %0.3f%%\n\n', ...
            evm.Peak,evm.RMS);

        fprintf('  Decoded MAC Sequence Control field contents:\n');
        fprintf('    Sequence number: %d\n\n',packetSeq(pktInd));
    end

    % Update search index
    searchOffset = pktOffset+double(indNonHTData(2));
    
    % Finish processing when a duplicate packet is detected. The
    % recovered data includes bits from duplicate frame
    % Remove the data bits from the duplicate frame
    if length(unique(packetSeq)) < length(packetSeq)
        rxBit = rxBit(1:length(unique(packetSeq)));
        packetSeq = packetSeq(1:length(unique(packetSeq)));
        break
    end

    pktInd = pktInd+1;
end

% Show final constellation
if (display_constellation_diagram)
        constellation(reshape(eqSym,[],1)); % Current constellation
end
%%
% Packet Decode Rate

packet_decode_counter = 0;

for i = 1:length(packetSeq)

    if packetSeq(i) == i-1
        packet_decode_counter = packet_decode_counter + 1;
    end

end

packet_decode_rate = (packet_decode_counter/length(packetSeq))*100;

fprintf("\nPacket Decode Counter: %d/%d \n", packet_decode_counter, length(packetSeq));
fprintf("Packet Decode Rate: %f \n", packet_decode_rate);


%%

if ~(isempty(fineTimingOffset) || isempty(pktOffset))

    rxData = cell2mat(rxBit);

    % Remove duplicate packets if any. Duplicate packets are located at the
    % end of rxData
    if length(packetSeq)>numMSDUs
        numDupPackets = size(rxData,2)-numMSDUs;
        rxData = rxData(:,1:end-numDupPackets);
    end

    % Initialize variables for while loop
    startSeq = [];
    i=-1;

    % Only execute this if one of the packet sequence values have been decoded
    % accurately
    if any(packetSeq<numMSDUs)
        while isempty(startSeq)
            % This searches for a known packetSeq value
            i = i + 1;
            startSeq = find(packetSeq==i);
        end
        % Circularly shift data so that received packets are in order for image reconstruction. It
        % is assumed that all packets following the starting packet are received in
        % order as this is how the image is transmitted.
        rxData = circshift(rxData,[0 -(startSeq(1)-i-1)]); % Order MAC fragments

        % Perform bit error rate (BER) calculation on reordered data
        bitErrorRate = comm.ErrorRate;
        err = bitErrorRate(double(rxData(:)), ...
            txDataBits(1:length(reshape(rxData,[],1))));
        fprintf('  \nBit Error Rate (BER):\n');
        fprintf('          Bit Error Rate (BER) = %0.5f\n',err(1));
        fprintf('          Number of bit errors = %d\n',err(2));
        fprintf('    Number of transmitted bits = %d\n\n',length(txDataBits));
    end

    decData = bi2de(reshape(rxData(:),8,[])');

    % Append NaNs to fill any missing image data
    if length(decData)<length(txImage)
        numMissingData = length(txImage)-length(decData);
        decData = [decData;NaN(numMissingData,1)];
    else
        decData = decData(1:length(txImage));
    end

    if (display_tx_rx_image)

        % Recreate image from received data
        fprintf('\nConstructing image from received data.\n');
        receivedImage = uint8(reshape(decData,imsize));

        % Plot received image 
        if exist('imFig','var') && ishandle(imFig) % If Tx figure is open
            figure(imFig); subplot(212);
        else
            figure; subplot(212);
        end

        imshow(receivedImage);
        title(sprintf('Received Image'));

    end
end


