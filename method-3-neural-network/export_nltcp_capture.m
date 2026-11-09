function export_nltcp_capture(sourceDir, outputDir, captures)
% Export aligned symbol streams without writing into the source data tree.
if nargin < 3
    captures = 1:19;
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
addpath(sourceDir);

datano = 51200;
upsampleno = 4;
NLTCP_qamdata = readmatrix(fullfile(sourceDir, 'v1.txt')) + ...
    1i * readmatrix(fullfile(sourceDir, 'v2.txt'));
decimal_PAM4_1 = readmatrix(fullfile(sourceDir, 'decimal_PAM4_1.txt'));
decimal_PAM4_2 = readmatrix(fullfile(sourceDir, 'decimal_PAM4_2.txt'));
dataout = readmatrix(fullfile(sourceDir, 'txdata', 'RF0.105_data36QAM.txt'));

for capture = captures
    rx1 = readmatrix(fullfile(sourceDir, 'rxdata', sprintf('rx1_%d.txt', capture)));
    rx2 = readmatrix(fullfile(sourceDir, 'rxdata', sprintf('rx2_%d.txt', capture)));

    [wave1, ~, ~, ~] = LMS_volterra_1DownS_Testnan(33, 0.022, 17, 0.00008, ...
        10000, rx1, dataout);
    cap1 = CAPdemodulation(wave1, upsampleno, datano);
    [symbol1, ~, ~, ~] = LMS_1DownS_Testnan(115, 0.0021, 10000, ...
        cap1, NLTCP_qamdata);

    [wave2, ~, ~, ~] = LMS_volterra_1DownS_Testnan(43, 0.013, 13, 0.00005, ...
        10000, rx2, dataout);
    cap2 = CAPdemodulation(wave2, upsampleno, datano);
    [symbol2, ~, ~, ~] = LMS_1DownS_Testnan(115, 0.0011, 10000, ...
        cap2, NLTCP_qamdata);

    aligned_path = fullfile(outputDir, sprintf('aligned_case%02d.mat', capture));
    datarx1 = cap1;
    datarx2 = cap2;
    save(aligned_path, 'datarx1', 'datarx2', 'symbol1', 'symbol2', ...
        'NLTCP_qamdata', 'decimal_PAM4_1', 'decimal_PAM4_2', '-v7');
    fprintf('capture %02d exported to %s\n', capture, aligned_path);
end
end
