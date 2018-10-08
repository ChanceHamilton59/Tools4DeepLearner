% autoEncoderImPatchCompLin.m
% 
% Author:       Chris DiMattina (cdimattina@fgcu.edu)
%
% Description:  This program demonstrates/implements a simple sparse 
%               auto-encoder compressing 12x12 image patches. 
%
%                
function autoEncoderImPatchSparseLin
    
    % paths
    addpath .\TLBP\
    trainPath = 'G:\TRNATFIROR\';
    
    % training set properties 
    nTrainSet   = 20;   % number of training sets
    XCell       = cell(1,nTrainSet); 
    imPerSet    = 4000; % images in each training set
    imSz        = 8;   % square dimension of image patches
 
    % problem parameters
    nBat        = 3000;  
    batSize     = 1000; % must be less thank imPerSet
    nIterBat    = 10;   
    eps         = 5e-03; 
    beta        = 1e-04; 
    lambda      = 0; % 1e-05;
    scl         = 1e-01;
       
    % define a neural network
    n_i         = imSz^2;   
    n_h         = n_i;    
    n_o         = n_i;      
    
    % initialize weights
    a.W1        = scl*(randn(n_h,n_i));
    a.W2        = a.W1'; %0.01*scl*(randn(n_o,n_h));
    a.b         = zeros(n_h,1); 
    a.b0        = zeros(n_o,1);
    a.gtype     = 'lin'; 
    
    errPlot     = NaN*ones(nBat*nIterBat,1);
    sparsePlot  = NaN*ones(nBat*nIterBat,1); 
    likValid    = NaN*ones(nBat*nIterBat,1); 
    
    % load up training data 
    for i=1:nTrainSet
        load(strcat(trainPath,sprintf('TrainSetPatch_%d.mat',i)) );
        XCell{i} = X;
        clear X; 
    end
    
    XVal   = XCell{nTrainSet}(:,1:batSize);
    TVal   = XVal;
    
    
    nPlotVal = 36; 
    figure(1); subplot(2,2,3);
    imgb(XVal(:,1:nPlotVal)); colormap('gray'); axis square;
    title('original');
    pause(0.1); 
    
    
    k = 1; 
    % main loop
    for i=1:nBat
        
        fn     = randi(nTrainSet-1);  % hold back last set as validation
        
        % Get batch of data
        batInd = randperm(imPerSet);
        batInd = batInd(1:batSize); 
        Xbat   = XCell{fn}(:,batInd); 
        Tbat   = Xbat;

        % For each batch perform gradient descent
        for j=1:nIterBat
            [Ybat,Zbat,gW1,gW2,~,~] = nnet3L(Xbat,Tbat,a);
            [YVal,~,~,~,~,~]        = nnet3L(XVal,TVal,a);
            % Sparsity constraints
            %[s,rho,gS]      = sparseActConKL(Xbat,a,0.05);
            [s,gS]          = sparseActConL1(Xbat,a);
            [~,gSW1]        = sparseWtConL2(a.W1);
            [~,gSW2]        = sparseWtConL2(a.W2);
            
           
           sparsePlot(k) = s; 
            
            % check to make sure gradient is reasonable
            if(sum(sum(isnan(gW1)))==0 && sum(sum(isnan(gW2)))==0)                 
                a.W1 = a.W1 - eps*gW1  +  beta*gS + lambda*gSW1;              
                a.W2 = a.W1'; % a.W2 - eps*gW2  +  lambda*gSW2;


                


                errPlot(k)  = sum(sum((Ybat-Tbat).^2));
                likValid(k)  = -sum(sum((YVal-TVal).^2));
            else
               if(k>1)
                   errPlot(k) = errPlot(k-1); 
                   likValid(k) = likValid(k-1); 
               end    
            end

            k = k + 1; 
            
            
        end
        
        
        figure(1); 
            subplot(2,2,1);
            semilogy(1:(nBat*nIterBat),-1*errPlot,'b'); axis square; hold on;
            %plot(1:(nBat*nIterBat),likValid,'r'); axis square; 
            ylabel('likelihood (tr/val)');
            xlabel('iteration'); 
          
            subplot(2,2,2);
            semilogy(1:(nBat*nIterBat),sparsePlot,'g'); axis square; 
            ylabel('sparsity');
            xlabel('iteration'); 
            pause(0.1); 
        
        
        
        figure(1); subplot(2,2,4);
        imgb(YVal(:,1:nPlotVal)); colormap('gray'); axis square;
        title('reconstructed');
 
%         figure(1); subplot(2,2,3);
%         hist(sum(a.W1.^2),20); axis square; 
%         title('W1 norms')
%         pause(0.1);
        
        figure(2);
        %subplot(1,2,1); 
        imgb(a.W1',0); colormap('gray'); axis square; 
        
%         subplot(1,2,2);
%         hist(Zbat,20); axis square; 
        
        
        
        
    end
    
    save('sparseLinTrained_out.mat','a')
    
end
