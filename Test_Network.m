
%%% test the model performance


% clear; clc;
format compact;

addpath(fullfile('data')); % Add path to Data and 
folderTest  = fullfile('data','CTTest'); %%% test dataset location
%folderTest  = fullfile('data','Test','Set68'); %%% test dataset

useGPU      = 1;

modelName   = 'CT_Model_10mas'; % Specify the model name
epoch       = 50; % Speificy the model at which epoch to use

%Loading the trained model 

load(fullfile('data',modelName,[modelName,'-epoch-',num2str(epoch),'.mat']));
net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);

%%%
net = vl_simplenn_tidy(net);


if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

% Read all the phantom data persent in test folder. 
ext         =  {'*.mat'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths)); % Computethe metric of performance. 
SSIMs = zeros(1,length(filePaths));

for i = 1:length(filePaths)
    load(fullfile(folderTest,filePaths(i).name));
OutVolume=zeros(size(CBF0_TSVD));

    for sliceno=1:size(CBF0_TSVD,3)
        
        label=single(CBF0_TSVD(:,:,sliceno)); % High Dose CBF map
        label = im2double(label);
        label_gt = single(CBF_GT(:,:,sliceno)); % Ground truth image
        input = single(CBF_TSVD(:,:,sliceno));  %  CBF low dose image 

        %%% convert to GPU
        if useGPU
            input = gpuArray(input);
        end

        res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
        output = input - res(end).x;

        %%% convert to CPU
        if useGPU
            output = gather(output);
            input  = gather(input);
        end
        OutVolume(:,:,sliceno)=output;
    end

        save(fullfile(folderTest,'results',['Denoised_',filePaths(i).name]),'OutVolume'); %saving denoised volume
end

disp([mean(PSNRs);mean(PSNRsIN)]);




