%% Designable Generative Adversarial Network (DGAN) %%
clc; clear all; close all; warning off;
addpath("subFile","savedData");

%% Load Input File
data_tag = date;
fileName = strcat('Y_by_F_uncer_', data_tag, '.xlsx');  % ★ Real 데이터 파일 불러오기
dataSet = xlsread(fileName,1,"A1:AD1086");  % Real 데이터 엑셀 범위 지정 % 셀AD까지면 15개, T까지면 10개, J까지면 5개 샘플 입력

%% Data Rearrange
dataSize = size(dataSet);

for i = 1:dataSize(1,2)/2
    data4D(:,:,1,i) = [dataSet(:,2*i)];
end

for i = 1:dataSize(1,2)/2
    data_Y1(:,i) = [dataSet(:,2*i-1)];
end
% 
for i = 1:dataSize(1,2)/2
    data_Y2(:,i) = [dataSet(:,2*i)];
end

augimds = augmentedImageDatastore([dataSize(1,1) 1],data4D);

%% Define Generator Network
filterSize = [36 1];  % 필터 크기 (높이, 너비) | CNNstructure 참고

latentDim = 5;  % 잠재변수 차원, 입출력 설계변수 개수

layersGenerator = [
    imageInputLayer([1 1 latentDim],'Normalization','none','Name','in')  % Input Layer: 잠재변수를 입력으로 하여 입력 층을 통과할 때마다 데이터 정규화 진행을 안하고, 이름이 'in'인 입력 층 생성
    
    transposedConv2dLayer(filterSize,64,'Name','tconv1')  % Transposed Convolution Layer 1
    batchNormalizationLayer('Name','bn1')  % Batch Layer 1: Convolution 신경망의 훈련속도를 높이기 위해 Convolution 층과 비선형 층(ReLU) 사이에 배치 정규화 층을 추가
    reluLayer('Name','relu1')  % Rectified Linear Unit Layer 1: 비선형 함수로써 입력값이 0보다 작으면 0으로, 0보다 크거나 같으면 실제 입력값 그대로 출력
    
    transposedConv2dLayer(filterSize,32,'Stride',2,'Cropping',0,'Name','tconv2')  % Transposed Convolution Layer 2
    batchNormalizationLayer('Name','bn2')  % Batch Layer 2
    reluLayer('Name','relu2')  % Rectified Linear Unit Layer 2
    
    transposedConv2dLayer(filterSize,16,'Stride',2,'Cropping',0,'Name','tconv3')  % Transposed Convolution Layer 3
    batchNormalizationLayer('Name','bn3')  % Batch Layer 3
    reluLayer('Name','relu3')  % Rectified Linear Unit Layer 3
    
    transposedConv2dLayer(filterSize,8,'Stride',2,'Cropping',0,'Name','tconv4')  % Transposed Convolution Layer 4
    batchNormalizationLayer('Name','bn4')  % Batch Layer 4
    reluLayer('Name','relu4')  % Rectified Linear Unit Layer 4
    
    transposedConv2dLayer(filterSize,1,'Stride',2,'Cropping',0,'Name','tconv5')  % Transposed Convolution Layer 5
    tanhLayer('Name','tanh')  % Hyperbolic Tangent Layer
    
    ];

lgraphGenerator = layerGraph(layersGenerator);  % 딥러닝을 위한 네트워크 계층의 그래프 객체 생성
dlnetGenerator = dlnetwork(lgraphGenerator);  % 딥러닝 네트워크로 변환

% % 전이학습 관련
% load('dlnetGenerator_TL.mat');
% dlnetGenerator = dlnetGenerator;

%% Define Discriminator Network
dropoutProb = 0.5;  % 드롭아웃 확률
scale = 0.2;  % leakyReLU 스케일 파라미터

layersDiscriminator = [
    
    imageInputLayer([dataSize(1,1) 1 1],'Normalization','none','Name','in')  % Input Layer: 실제 데이터를 입력으로 하여 입력 층을 통과할 때마다 데이터 정규화 진행을 안하고, 이름이 'in'인 입력 층 생성
    
    dropoutLayer(0.5,'Name','dropout')  % Dropout Layer
    
    convolution2dLayer(filterSize,8,'Stride',2,'Padding',0,'Name','conv1')  % Convolution Layer 1
    leakyReluLayer(scale,'Name','lrelu1')  % Leaky ReLU 1: 비선형 함수로써 입력값이 0보다 작으면 scale에 실제 입력값을 곱하고, 0보다 크거나 같으면 실제 입력값 그대로 출력
    
    convolution2dLayer(filterSize,16,'Stride',2,'Padding',0,'Name','conv2')  % Convolution Layer 2
    batchNormalizationLayer('Name','bn2')  % Batch Layer 2
    leakyReluLayer(scale,'Name','lrelu2')  % Leaky ReLU 2
    
    convolution2dLayer(filterSize,32,'Stride',2,'Padding',0,'Name','conv3')  % Convolution Layer 3
    batchNormalizationLayer('Name','bn3')  % Batch Layer 3
    leakyReluLayer(scale,'Name','lrelu3')  % Leaky ReLU 3
    
    convolution2dLayer(filterSize,64,'Stride',2,'Padding',0,'Name','conv4')  % Convolution Layer 4
    batchNormalizationLayer('Name','bn4')  % Batch Layer 4
    leakyReluLayer(scale,'Name','lrelu4')  % Leaky ReLU 4
    
    convolution2dLayer(filterSize,1,'Stride',2,'Padding',0,'Name','conv5')  % Convolution Layer 5
    
    ];

lgraphDiscriminator = layerGraph(layersDiscriminator);  % 딥러닝을 위한 네트워크 계층의 그래프 객체 생성
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);  % 딥러닝 네트워크로 변환

% load('dlnetDiscriminator_TL.mat');
% dlnetDiscriminator = dlnetDiscriminator;



%% Define Inverse Generator Network
layersInverseGenerator = [
    
    imageInputLayer([dataSize(1,1) 1 1],'Normalization','none','Name','in')  % Input Layer: 잠재변수를 입력으로 하여 입력 층을 통과할 때마다 데이터 정규화 진행을 안하고, 이름이 'in'인 입력 층 생성
 
    convolution2dLayer(filterSize,8,'Stride',2,'Padding',0,'Name','conv1')  % Convolution Layer 1
    
    convolution2dLayer(filterSize,16,'Stride',2,'Padding',0,'Name','conv2')  % Convolution Layer 2
    
    convolution2dLayer(filterSize,32,'Stride',2,'Padding',0,'Name','conv3')  % Convolution Layer 3
    
    convolution2dLayer(filterSize,64,'Stride',2,'Padding',0,'Name','conv4')  % Convolution Layer 4
    
    convolution2dLayer(filterSize,latentDim,'Stride',2,'Padding',0,'Name','conv5')  % Convolution Layer 5  
    
    ];

lgraphInverseGenerator = layerGraph(layersInverseGenerator);  % 딥러닝을 위한 네트워크 계층의 그래프 객체 생성
dlnetInverseGenerator = dlnetwork(lgraphInverseGenerator);  % 딥러닝 네트워크로 변환

%% Specify Training Options
epoch = 1000;  % ★ 알고리즘 반복 횟수 선정
miniBatchSize = 5; % 예전 영민형 코드에서는 37, 배치 크기 선정 (전체 데이터 수의 20 %)
augimds.MiniBatchSize = miniBatchSize;
learnRateGenerator = 0.001;  % 생성기 학습률
learnRateDiscriminator = 0.001;  % 판별기 학습률: 만약 판별기의 학습률이 크다면 생성된 이미지에 대해 진짜인지 가짜인지 빠르게 판별하는 쪽으로 학습되므로 생성기가 제대로 작동하지 않을 수 있음
learnRateGeneratorAndInverseGenerator = 0.001; % 역생성기 학습률
trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;
avgGradientsGenerator = [];
avgGradientsSquaredGenerator = [];
avgGradientsInverseGenerator = [];
avgGradientsSquaredInverseGenerator = [];


%% Train Model
global probReal probGenerated scoreGenerator scoreDiscriminator lossInverseGenerator

f = figure;
f.Position(3) = 3*f.Position(3);
scoreAxes = subplot(1,3,2);
lineScoreGenerator = animatedline(scoreAxes,'Color',[0 0.447 0.741]);
lineScoreDiscriminator = animatedline(scoreAxes,'Color',[0.85 0.325 0.098]);
legend('Generator','Discriminator');
% ylim([0 1])
set(gca,'fontsize',15,'fontname','times new roman');
xlabel("Iteration",'fontsize',18,'fontname','times new roman')
ylabel("Score",'fontsize',18,'fontname','times new roman')
grid on
lossAxes = subplot(1,3,3);
lineLossInverseGenerator = animatedline(lossAxes,'Color',[1 0 0]);
legend('InverseGenerator');
set(gca,'fontsize',15,'fontname','times new roman');
xlabel("Iteration",'fontsize',18,'fontname','times new roman')
ylabel("Loss",'fontsize',18,'fontname','times new roman')
grid on

iteration = 0;
start = tic;
Result_GeneratorScore = [];
Result_DiscriminatorScore = [];
Result_InverseGeneratorLoss = [];

% Loop over epochs
for i = 1:epoch      
          
    % Reset and shuffle datastore
    reset(augimds);  % 데이터 저장소를 아무것도 읽지 않은 상태로 초기화
    augimds = shuffle(augimds);  % 데이터 순서를 섞음
    
    % Loop over mini-batches
    while hasdata(augimds)
        iteration = iteration + 1;

        % Read mini-batch of data
        data = read(augimds);
        
        % Ignore last partial mini-batch of epoch
        if size(data,1) < miniBatchSize
            continue
        end
        
        % Concatenate mini-batch of data and generate latent inputs for the generator network
        Y_Training = cat(4,data{:,1}{:});  % 배열 결합 cat(dim,A1,A2,...): 차원 dim을 따라 A1,A2,...를 결합
        X1_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x1
        X2_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x2
        X3_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x3
        X4_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x4
        X5_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x5
%         X6_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x6
%         X7_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x7
%         X8_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x8
%         X9_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x9
%         X10_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x10
%         X11_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x11
%         X12_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x12
%         X13_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x13
%         X14_Training = randn(1,1,1,size(Y_Training,4));  % Design variable x14
        
        for j = 1:size(Y_Training,4)
            X_Training(:,:,1,j) = [X1_Training(:,:,1,j)];
            X_Training(:,:,2,j) = [X2_Training(:,:,1,j)];
            X_Training(:,:,3,j) = [X3_Training(:,:,1,j)];
            X_Training(:,:,4,j) = [X4_Training(:,:,1,j)];
            X_Training(:,:,5,j) = [X5_Training(:,:,1,j)];
%             X_Training(:,:,6,j) = [X6_Training(:,:,1,j)];
%             X_Training(:,:,7,j) = [X7_Training(:,:,1,j)];
%             X_Training(:,:,8,j) = [X8_Training(:,:,1,j)];
%             X_Training(:,:,9,j) = [X9_Training(:,:,1,j)];
%             X_Training(:,:,10,j) = [X10_Training(:,:,1,j)];
%             X_Training(:,:,11,j) = [X11_Training(:,:,1,j)];
%             X_Training(:,:,12,j) = [X12_Training(:,:,1,j)];
%             X_Training(:,:,13,j) = [X13_Training(:,:,1,j)];
%             X_Training(:,:,14,j) = [X14_Training(:,:,1,j)];
        end
        
        % Normalize the data
        min_Y1 = min(data_Y1(:)); max_Y1 = max(data_Y1(:));
        min_Y2 = min(data_Y2(:)); max_Y2 = max(data_Y2(:));
        scale_min_Y = -1; scale_max_Y = 1;
        
        for j = 1:miniBatchSize
            Y_Training_Normalize(:,:,1,j) = [(scale_max_Y*(Y_Training(:,1,1,j)-min_Y2)+scale_min_Y*(max_Y2-Y_Training(:,1,1,j)))/(max_Y2-min_Y2)];
        end
                
        % Convert mini-batch of data to dlarray specify the dimension labels 'SSCB' (spatial, spatial, channel, batch)
        dlY_Training = dlarray(Y_Training_Normalize,'SSCB');
        dlX_Training = dlarray(X_Training,'SSCB');
        
        % Evaluate the model gradients and the generator state using dlfeval and the modelGradients function
        [gradientsGenerator,gradientsDiscriminator,stateGenerator,scoreGenerator,scoreDiscriminator] = ...
            dlfeval(@modelGradients,dlnetGenerator,dlnetDiscriminator,dlY_Training,dlX_Training);
        dlnetGenerator.State = stateGenerator;
        
        % Update the discriminator network parameters
        [dlnetDiscriminator.Learnables,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator.Learnables,gradientsDiscriminator,trailingAvgDiscriminator,trailingAvgSqDiscriminator,iteration,learnRateDiscriminator,gradientDecayFactor,squaredGradientDecayFactor);
        
        % Update the generator network parameters
        [dlnetGenerator.Learnables,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator.Learnables,gradientsGenerator,trailingAvgGenerator,trailingAvgSqGenerator,iteration,learnRateGenerator,gradientDecayFactor,squaredGradientDecayFactor);
        
        % Evaluate the inverse model gradients
        [infGrad,genGrad] = ...
            dlfeval(@modelGradients2,dlnetGenerator,dlnetInverseGenerator,dlX_Training);
        
        % Update the inverse generator network parameters
        [dlnetInverseGenerator.Learnables,avgGradientsInverseGenerator,avgGradientsSquaredInverseGenerator] = ...
            adamupdate(dlnetInverseGenerator.Learnables,genGrad,avgGradientsInverseGenerator,avgGradientsSquaredInverseGenerator,iteration,learnRateGeneratorAndInverseGenerator);
        
        % Update the generator network parameters
        [dlnetGenerator.Learnables,avgGradientsGenerator,avgGradientsSquaredGenerator] = ...
            adamupdate(dlnetGenerator.Learnables,infGrad,avgGradientsGenerator,avgGradientsSquaredGenerator,iteration,learnRateGeneratorAndInverseGenerator);
        
        % Every 1 iterations, display batch of generated images using the held-out generator input
        if mod(iteration,1) == 0 || iteration == 1
            
            % Generate data using the held-out generator input
            dlY_Training_Generated = predict(dlnetGenerator,dlX_Training);
            dlX_Training_InverseGenerated = predict(dlnetInverseGenerator,dlY_Training_Generated);
            
            % Rescale the data and display the data                  
%             Y1_Training_Normalize = extractdata(dlY_Training_Generated(:,1));
            Y2_Training_Normalize = extractdata(dlY_Training_Generated(:,1));
            Y_Training_DeNormalize = [((Y2_Training_Normalize-scale_min_Y)*max_Y2+(scale_max_Y-Y2_Training_Normalize)*min_Y2)/(scale_max_Y-scale_min_Y)];
            
            X1_Training_Normalize = [];
            X2_Training_Normalize = [];
            X3_Training_Normalize = [];
            X4_Training_Normalize = [];
            X5_Training_Normalize = [];
%             X6_Training_Normalize = [];
%             X7_Training_Normalize = [];
%             X8_Training_Normalize = [];
%             X9_Training_Normalize = [];
%             X10_Training_Normalize = [];
%             X11_Training_Normalize = [];
%             X12_Training_Normalize = [];
%             X13_Training_Normalize = [];
%             X14_Training_Normalize = [];

            for k = 1:miniBatchSize    
                X1_Training_Normalize = [X1_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,1,k))];
                X2_Training_Normalize = [X2_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,2,k))];
                X3_Training_Normalize = [X3_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,3,k))];
                X4_Training_Normalize = [X4_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,4,k))];
                X5_Training_Normalize = [X5_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,5,k))];
%                 X6_Training_Normalize = [X6_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,6,k))];
%                 X7_Training_Normalize = [X7_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,7,k))];
%                 X8_Training_Normalize = [X8_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,8,k))];
%                 X9_Training_Normalize = [X9_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,9,k))];
%                 X10_Training_Normalize = [X10_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,10,k))];
%                 X11_Training_Normalize = [X11_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,11,k))];
%                 X12_Training_Normalize = [X12_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,12,k))];
%                 X13_Training_Normalize = [X13_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,13,k))];
%                 X14_Training_Normalize = [X14_Training_Normalize extractdata(dlX_Training_InverseGenerated(:,:,14,k))];
            end

            % % 밑에(generate new data)도 똑같은 게 잇네
            % % ★ 설계변수 범위 (1)
            min_X1 = 4; max_X1 = 6;
            min_X2 = 2.4; max_X2 = 3.6;
            min_X3 = 7.2; max_X3 = 10.8;
            min_X4 = 2; max_X4 = 3;
            min_X5 = 39.2; max_X5 = 58.8;
%             min_X6 = 530; max_X6 = 640;
%             min_X7 = 530; max_X7 = 640;
%             min_X8 = 420; max_X8 = 500;
%             min_X9 = 590; max_X9 = 700;
%             min_X10 = 530; max_X10 = 640;
%             min_X11 = 4166.7; max_X11 = 4444.4;
%             min_X12 = 0.3; max_X12 = 0.5;
%             min_X13 = 8; max_X13 = 10;
%             min_X14 = 0; max_X14 = 50;
            scale_min_X = -3; scale_max_X = 3;

            X1_Training_DeNormalize = [((X1_Training_Normalize-scale_min_X)*max_X1+(scale_max_X-X1_Training_Normalize)*min_X1)/(scale_max_X-scale_min_X)]; 
            X2_Training_DeNormalize = [((X2_Training_Normalize-scale_min_X)*max_X2+(scale_max_X-X2_Training_Normalize)*min_X2)/(scale_max_X-scale_min_X)];
            X3_Training_DeNormalize = [((X3_Training_Normalize-scale_min_X)*max_X3+(scale_max_X-X3_Training_Normalize)*min_X3)/(scale_max_X-scale_min_X)]; 
            X4_Training_DeNormalize = [((X4_Training_Normalize-scale_min_X)*max_X4+(scale_max_X-X4_Training_Normalize)*min_X4)/(scale_max_X-scale_min_X)]; 
            X5_Training_DeNormalize = [((X5_Training_Normalize-scale_min_X)*max_X5+(scale_max_X-X5_Training_Normalize)*min_X5)/(scale_max_X-scale_min_X)]; 
%             X6_Training_DeNormalize = [((X6_Training_Normalize-scale_min_X)*max_X6+(scale_max_X-X6_Training_Normalize)*min_X6)/(scale_max_X-scale_min_X)]; 
%             X7_Training_DeNormalize = [((X7_Training_Normalize-scale_min_X)*max_X7+(scale_max_X-X7_Training_Normalize)*min_X7)/(scale_max_X-scale_min_X)]; 
%             X8_Training_DeNormalize = [((X8_Training_Normalize-scale_min_X)*max_X8+(scale_max_X-X8_Training_Normalize)*min_X8)/(scale_max_X-scale_min_X)]; 
%             X9_Training_DeNormalize = [((X9_Training_Normalize-scale_min_X)*max_X9+(scale_max_X-X9_Training_Normalize)*min_X9)/(scale_max_X-scale_min_X)]; 
%             X10_Training_DeNormalize = [((X10_Training_Normalize-scale_min_X)*max_X10+(scale_max_X-X10_Training_Normalize)*min_X10)/(scale_max_X-scale_min_X)]; 
%             X11_Training_DeNormalize = [((X11_Training_Normalize-scale_min_X)*max_X11+(scale_max_X-X11_Training_Normalize)*min_X11)/(scale_max_X-scale_min_X)]; 
%             X12_Training_DeNormalize = [((X12_Training_Normalize-scale_min_X)*max_X12+(scale_max_X-X12_Training_Normalize)*min_X12)/(scale_max_X-scale_min_X)]; 
%             X13_Training_DeNormalize = [((X13_Training_Normalize-scale_min_X)*max_X13+(scale_max_X-X13_Training_Normalize)*min_X13)/(scale_max_X-scale_min_X)]; 
%             X14_Training_DeNormalize = [((X14_Training_Normalize-scale_min_X)*max_X14+(scale_max_X-X14_Training_Normalize)*min_X14)/(scale_max_X-scale_min_X)];
            
            figure(1)
            subplot(1,3,1)
            plot(data_Y1(:,1),Y_Training_DeNormalize(:,1),'k');
            set(gca,'fontsize',15,'fontname','times new roman');
            xlabel('y_{1}','fontsize',18,'fontname','times new roman');
            ylabel('y_{2}','fontsize',18,'fontname','times new roman');
%             xlim([0 200]); 
%             ylim([0 160]);
            
            % Update the title with training progress information
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
            
            % Update the scores plot
            subplot(1,3,2)            
            addpoints(lineScoreGenerator,iteration,double(gather(extractdata(scoreGenerator))));            
            addpoints(lineScoreDiscriminator,iteration,double(gather(extractdata(scoreDiscriminator))));
            
            % Update the title with training progress information
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
            
            % Update the loss plot
            subplot(1,3,3)            
            addpoints(lineLossInverseGenerator,iteration,double(gather(extractdata(lossInverseGenerator)))); 
                        
            % Update the title with training progress information
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            title(...
                "Epoch: " + i + ", " + ...
                "Iteration: " + iteration + ", " + ...
                "Elapsed: " + string(D))
            
            drawnow                        
        end
    end
    
    ProbabilityReal = double(gather(extractdata(mean(probReal))));
    ProbabilityFake = double(gather(extractdata(mean(probGenerated))));
    GeneratorScore = double(gather(extractdata(scoreGenerator)));
    DiscriminatorScore = double(gather(extractdata(scoreDiscriminator)));
    InverseGeneratorLoss = double(gather(extractdata(lossInverseGenerator)));
    
    Result_GeneratorScore = [Result_GeneratorScore; GeneratorScore];
    Result_DiscriminatorScore = [Result_DiscriminatorScore; DiscriminatorScore];
    Result_InverseGeneratorLoss = [Result_InverseGeneratorLoss; InverseGeneratorLoss];
    
    disp("Epoch: " + i + ", " + "ProbabilityReal: " + ProbabilityReal + ", " + "ProbabilityFake: " + ProbabilityFake + ", " + "GeneratorScore: " + GeneratorScore + ", " + "DiscriminatorScore: " + DiscriminatorScore + ", " + "InverseGeneratorLoss: " + InverseGeneratorLoss);
    
%     if (0.495 <= ProbabilityReal) & (ProbabilityReal <= 0.505) & (0.495 <= ProbabilityFake) & (ProbabilityFake <= 0.505)
%         break
%     end   
end


%% Generate New Data
numGenerationData = 300;  % ★ 생성할 잠재변수 데이터셋 수 (= 생성 데이터 수)
X1_Test = randn(1,1,1,numGenerationData);  % Design variable x1
X2_Test = randn(1,1,1,numGenerationData);  % Design variable x2
X3_Test = randn(1,1,1,numGenerationData);  % Design variable x3
X4_Test = randn(1,1,1,numGenerationData);  % Design variable x4
X5_Test = randn(1,1,1,numGenerationData);  % Design variable x5
% X6_Test = randn(1,1,1,numGenerationData);  % Design variable x6
% X7_Test = randn(1,1,1,numGenerationData);  % Design variable x7
% X8_Test = randn(1,1,1,numGenerationData);  % Design variable x8
% X9_Test = randn(1,1,1,numGenerationData);  % Design variable x9
% X10_Test = randn(1,1,1,numGenerationData);  % Design variable x10
% X11_Test = randn(1,1,1,numGenerationData);  % Design variable x11
% X12_Test = randn(1,1,1,numGenerationData);  % Design variable x12
% X13_Test = randn(1,1,1,numGenerationData);  % Design variable x13
% X14_Test = randn(1,1,1,numGenerationData);  % Design variable x14

for j = 1:numGenerationData
    X_Test(:,:,1,j) = [X1_Test(:,:,1,j)];
    X_Test(:,:,2,j) = [X2_Test(:,:,1,j)];
    X_Test(:,:,3,j) = [X3_Test(:,:,1,j)];
    X_Test(:,:,4,j) = [X4_Test(:,:,1,j)];
    X_Test(:,:,5,j) = [X5_Test(:,:,1,j)];
%     X_Test(:,:,6,j) = [X6_Test(:,:,1,j)];
%     X_Test(:,:,7,j) = [X7_Test(:,:,1,j)];
%     X_Test(:,:,8,j) = [X8_Test(:,:,1,j)];
%     X_Test(:,:,9,j) = [X9_Test(:,:,1,j)];
%     X_Test(:,:,10,j) = [X10_Test(:,:,1,j)];
%     X_Test(:,:,11,j) = [X11_Test(:,:,1,j)];
%     X_Test(:,:,12,j) = [X12_Test(:,:,1,j)];
%     X_Test(:,:,13,j) = [X13_Test(:,:,1,j)];
%     X_Test(:,:,14,j) = [X14_Test(:,:,1,j)];
end

dlX_Test = dlarray(X_Test,'SSCB');

dlY_Test_Generated = predict(dlnetGenerator,dlX_Test);
dlX_Test_Generated = predict(dlnetInverseGenerator,dlY_Test_Generated);

for k = 1:numGenerationData
%     Y1_Test_Normalize = extractdata(dlY_Test_Generated(:,1,1,k));
    Y2_Test_Normalize = extractdata(dlY_Test_Generated(:,1,1,k));
    Y_Test_DeNormalize(:,:,1,k) = [((Y2_Test_Normalize-scale_min_Y)*max_Y2+(scale_max_Y-Y2_Test_Normalize)*min_Y2)/(scale_max_Y-scale_min_Y)];
end

Result_Y = [];

for k = 1:numGenerationData
    Result_Y = [Result_Y Y_Test_DeNormalize(:,:,1,k)];    
end

X1_Test_Normalize = [];
X2_Test_Normalize = [];
X3_Test_Normalize = [];
X4_Test_Normalize = [];
X5_Test_Normalize = [];
% X6_Test_Normalize = [];
% X7_Test_Normalize = [];
% X8_Test_Normalize = [];
% X9_Test_Normalize = [];
% X10_Test_Normalize = [];
% X11_Test_Normalize = [];
% X12_Test_Normalize = [];
% X13_Test_Normalize = [];
% X14_Test_Normalize = [];

for k = 1:numGenerationData    
    X1_Test_Normalize = [X1_Test_Normalize extractdata(dlX_Test_Generated(:,:,1,k))];
    X2_Test_Normalize = [X2_Test_Normalize extractdata(dlX_Test_Generated(:,:,2,k))];
    X3_Test_Normalize = [X3_Test_Normalize extractdata(dlX_Test_Generated(:,:,3,k))];
    X4_Test_Normalize = [X4_Test_Normalize extractdata(dlX_Test_Generated(:,:,4,k))];
    X5_Test_Normalize = [X5_Test_Normalize extractdata(dlX_Test_Generated(:,:,5,k))];
%     X6_Test_Normalize = [X6_Test_Normalize extractdata(dlX_Test_Generated(:,:,6,k))];
%     X7_Test_Normalize = [X7_Test_Normalize extractdata(dlX_Test_Generated(:,:,7,k))];
%     X8_Test_Normalize = [X8_Test_Normalize extractdata(dlX_Test_Generated(:,:,8,k))];
%     X9_Test_Normalize = [X9_Test_Normalize extractdata(dlX_Test_Generated(:,:,9,k))];
%     X10_Test_Normalize = [X10_Test_Normalize extractdata(dlX_Test_Generated(:,:,10,k))];
%     X11_Test_Normalize = [X11_Test_Normalize extractdata(dlX_Test_Generated(:,:,11,k))];
%     X12_Test_Normalize = [X12_Test_Normalize extractdata(dlX_Test_Generated(:,:,12,k))];
%     X13_Test_Normalize = [X13_Test_Normalize extractdata(dlX_Test_Generated(:,:,13,k))];
%     X14_Test_Normalize = [X14_Test_Normalize extractdata(dlX_Test_Generated(:,:,14,k))];
end

% % ★ 설계변수 범위 (2)
min_X1 = 4; max_X1 = 6;
min_X2 = 2.4; max_X2 = 3.6;
min_X3 = 7.2; max_X3 = 10.8;
min_X4 = 2; max_X4 = 3;
min_X5 = 39.2; max_X5 = 58.8;
% min_X1 = 1.9; max_X1 = 2.1;
% min_X2 = 1.9; max_X2 = 2.1;
% min_X3 = 1.9; max_X3 = 2.1;
% min_X4 = 1.52; max_X4 = 1.68;
% min_X5 = 0.95; max_X5 = 1.05;
% min_X6 = 530; max_X6 = 640;
% min_X7 = 530; max_X7 = 640;
% min_X8 = 420; max_X8 = 500;
% min_X9 = 590; max_X9 = 700;
% min_X10 = 530; max_X10 = 640;
% min_X11 = 4166.7; max_X11 = 4444.4;
% min_X12 = 0.3; max_X12 = 0.5;
% min_X13 = 8; max_X13 = 10;
% min_X14 = 0; max_X14 = 50;
scale_min_X = -3; scale_max_X = 3;

Result_X1 = [((X1_Test_Normalize-scale_min_X)*max_X1+(scale_max_X-X1_Test_Normalize)*min_X1)/(scale_max_X-scale_min_X)]; 
Result_X2 = [((X2_Test_Normalize-scale_min_X)*max_X2+(scale_max_X-X2_Test_Normalize)*min_X2)/(scale_max_X-scale_min_X)]; 
Result_X3 = [((X3_Test_Normalize-scale_min_X)*max_X3+(scale_max_X-X3_Test_Normalize)*min_X3)/(scale_max_X-scale_min_X)]; 
Result_X4 = [((X4_Test_Normalize-scale_min_X)*max_X4+(scale_max_X-X4_Test_Normalize)*min_X4)/(scale_max_X-scale_min_X)]; 
Result_X5 = [((X5_Test_Normalize-scale_min_X)*max_X5+(scale_max_X-X5_Test_Normalize)*min_X5)/(scale_max_X-scale_min_X)]; 
% Result_X6 = [((X6_Test_Normalize-scale_min_X)*max_X6+(scale_max_X-X6_Test_Normalize)*min_X6)/(scale_max_X-scale_min_X)]; 
% Result_X7 = [((X7_Test_Normalize-scale_min_X)*max_X7+(scale_max_X-X7_Test_Normalize)*min_X7)/(scale_max_X-scale_min_X)]; 
% Result_X8 = [((X8_Test_Normalize-scale_min_X)*max_X8+(scale_max_X-X8_Test_Normalize)*min_X8)/(scale_max_X-scale_min_X)]; 
% Result_X9 = [((X9_Test_Normalize-scale_min_X)*max_X9+(scale_max_X-X9_Test_Normalize)*min_X9)/(scale_max_X-scale_min_X)]; 
% Result_X10 = [((X10_Test_Normalize-scale_min_X)*max_X10+(scale_max_X-X10_Test_Normalize)*min_X10)/(scale_max_X-scale_min_X)]; 
% Result_X11 = [((X11_Test_Normalize-scale_min_X)*max_X11+(scale_max_X-X11_Test_Normalize)*min_X11)/(scale_max_X-scale_min_X)]; 
% Result_X12 = [((X12_Test_Normalize-scale_min_X)*max_X12+(scale_max_X-X12_Test_Normalize)*min_X12)/(scale_max_X-scale_min_X)]; 
% Result_X13 = [((X13_Test_Normalize-scale_min_X)*max_X13+(scale_max_X-X13_Test_Normalize)*min_X13)/(scale_max_X-scale_min_X)]; 
% Result_X14 = [((X14_Test_Normalize-scale_min_X)*max_X14+(scale_max_X-X14_Test_Normalize)*min_X14)/(scale_max_X-scale_min_X)]; 

figure(2)
for i = 1:numGenerationData
    plot(data_Y1(:,1),Result_Y(:,i),'k'); hold on;
    set(gca,'fontsize',15,'fontname','times new roman');
    xlabel('y_{1}','fontsize',18,'fontname','times new roman');
    ylabel('y_{2}','fontsize',18,'fontname','times new roman');
    title("Generated Data",'fontsize',18,'fontname','times new roman')
end


%% Results
Result_Y = Result_Y;
Result_X1 = Result_X1';
Result_X2 = Result_X2';
Result_X3 = Result_X3';
Result_X4 = Result_X4';
Result_X5 = Result_X5';
% Result_X6 = Result_X6';
% Result_X7 = Result_X7';
% Result_X8 = Result_X8';
% Result_X9 = Result_X9';
% Result_X10 = Result_X10';
% Result_X11 = Result_X11';
% Result_X12 = Result_X12';
% Result_X13 = Result_X13';
% Result_X14 = Result_X14';


%% Model Gradients Function
function [gradientsGenerator,gradientsDiscriminator,stateGenerator,scoreGenerator,scoreDiscriminator] = ...
    modelGradients(dlnetGenerator,dlnetDiscriminator,Y,X)

global probReal probGenerated scoreGenerator scoreDiscriminator

% Calculate the predictions for real data with the discriminator network
YPred = forward(dlnetDiscriminator,Y);

% Calculate the predictions for generated data with the discriminator network
[dlYGenerated,stateGenerator] = forward(dlnetGenerator,X);
YPredGenerated = forward(dlnetDiscriminator,dlYGenerated);

% Convert the discriminator outputs to probabilities
probReal = sigmoid(YPred);
probGenerated = sigmoid(YPredGenerated);

% Calculate the score of the discriminator
scoreDiscriminator = ((mean(probReal)+mean(1-probGenerated))/2);

% Calculate the score of the generator
scoreGenerator = mean(probGenerated);

% Calculate the GAN loss
[lossGenerator,lossDiscriminator] = ganLoss(probReal,probGenerated);

% For each network, calculate the gradients with respect to the loss
gradientsGenerator = dlgradient(lossGenerator,dlnetGenerator.Learnables,'RetainData',true);
gradientsDiscriminator = dlgradient(lossDiscriminator,dlnetDiscriminator.Learnables);
end


%% GAN Loss Function
function [lossGenerator,lossDiscriminator] = ganLoss(probReal,probGenerated)

% Calculate the loss for the discriminator network
lossDiscriminator = -mean(log(probReal)) - mean(log(1-probGenerated));

% Calculate the loss for the generator network
lossGenerator = -mean(log(probGenerated));
end


%% Inverse Model Gradients Function
function [infGrad,genGrad] = modelGradients2(dlnetGenerator,dlnetInverseGenerator,X)

global lossInverseGenerator

YGenerated = forward(dlnetGenerator,X);
XPred = forward(dlnetInverseGenerator,YGenerated);
lossInverseGenerator = InverseGLoss(X,XPred);
[genGrad,infGrad] = dlgradient(lossInverseGenerator,dlnetInverseGenerator.Learnables,dlnetGenerator.Learnables);
end


%% Inverse Generator Loss Function
function lossInverseGenerator = InverseGLoss(X,XPred)
squares = 0.5*(XPred-X).^2;
reconstructionLoss  = sum(squares,[1,2,3]);
lossInverseGenerator = mean(reconstructionLoss);
end