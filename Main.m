%% SUBJECT-INDEPENDENT ERP-BASED BRAIN-COMPUTER INTERFACE
% Author: Kha Vo, voanhkha@yahoo.com.

% This software reproduces the results shown in Table III in the paper.
% Please read the instruction guide pdf file for editing the correct parameters.


clear all; close all
setpath
testpath = '';
learning_time = zeros(15,10);

%% Input parameters
method = 'SVM'; %SVM or Bayes
typedataset = 'Akimpech'; % Akimpech or ALS
fileclassifier=load('ENS-3'); % ENS-2, ENS-3, ENS-4, or ENS-6
theta = [1.5 .75];
Adaptive_flag = 0  ; % 1 for activate Adaptive Learning
max_adaptation = 4; % number of first letters to update classifiers

%% ----------------------------------------
%% AKIMPECH CLASSIFICATION
%----------------------------------------
switch typedataset
    case 'Akimpech'
subject={'FSZ' 'GCE' 'ICE' 'IZH'...
    'JLD' 'JLP' 'JMR' 'JSC' ...
            'JST' 'LAC' 'LAG' 'PGA' 'WFG' 'XCL'...
    }; 
 correct_result={'ROMACORALRELOJ' 'CASTABATCHROCA' 'AUTOCLAVEZETASHIELO' '1987JUN19YOUBIOMEDICA'...
                'HOLAFEOPAULA' 'LAPICEROLIBROBANCO' ...
                'PERROBARCOTIMON1' ...
                'GATOLPIZOTON' 'ZUKYMAYTEAZUL' 'LAURADANZACASA' ...
                'DORMIRQUIERO_COCAHAMBRE' ...
                             'TRIPTOFANOAGUAPAEL' 'UAM_IINGENIERIABIOMEDICA'...
                'GATOPEZPERRO' ...
                };
datafolder = 'TestCharacters/';           
 matrix=['ABCDEF';'GHIJKL';'MNOPQR';'STUVWX';'YZ1234';'56789_'];    
 max_iter = 15;
    case 'ALS'  
subject={'A01', 'A02' 'A03' 'A04' 'A05' 'A06' 'A07' 'A08'};
 correct_result={'2EZUPPAGATTBOMENTEVIOLAAREBUSCI5ROS','2EZUPPAGATTBOMENTEVIOLAAREBUSCI5ROS' ...
   '2EZUPPAGATTBOMENTEVIOLAAREBUSCI5ROS','2EZUPPAGATTBOMENTEVIOLAAREBUSCI5ROS' ...             
 '2EZUPPAGATTBOMENTEVIOLAAREBUSCI5ROS','2EZUPPAGATTBOMENTEVIOLAAREBUSCI5ROS' ...
 '2EZUPPAGATTBOMENTEVIOLAAREBUSCI5ROS','2EZUPPAGATTBOMENTEVIOLAAREBUSCI5ROS' ...
 };   
datafolder = 'D:\Google Drive\Ongoing Projects\EEG_data\preprocesseddata\ALS\'; 
  matrix=['ABCDEF';'GHIJKL';'MNOPQR';'STUVWX';'YZ1234';'56789_'];
  max_iter = 10;
end

% Extract target rows and columns of all characters of each subject and
% store them 
for i = 1:length(correct_result)
 for j = 1:length(correct_result{i})
    [r(i, j), c(i, j)] = find(matrix == correct_result{i}(j));
 end
end

iter_all = cell(length(subject), 1);
letter_all = cell(length(subject), 1);
s_row_all = cell(length(subject), 1);
s_col_all = cell(length(subject), 1);

nb_sub = 0;


s_row_target = []; s_col_target = [];
s_row_nontarget = [];
for sub = subject
   nb_sub = nb_sub + 1;
   adaptation = 0;
   sub = subject(nb_sub);
    %fprintf(['Subject ' char(sub) ' (' num2str(nb_sub) ')' '\n'])
    files_dir = dir([datafolder char(sub) '*.mat']); 
    
    for j=1:length(files_dir)
        filet{j} =files_dir(j).name;
    end
    
    classifiers = fileclassifier.classifier;
    
    nb_file = 0;
    for file = filet
    nb_file = nb_file+1;
    filetest = load([datafolder char(file)]);
    x = filetest.x; 
    code = filetest.code;
    
    switch method
        case 'Bayes'
    [iter, letter, row, col]  = Bayes_Dynamic_Classify_Letter(x, code, classifiers, typedataset);
        case 'SVM'
    [iter, letter, row, col, s_row_iter, s_col_iter]  = Dynamic_Classify_Letter(x, code, classifiers, typedataset, theta);       
    end
    
    iter_all{nb_sub} = [iter_all{nb_sub}; iter];
    letter_all{nb_sub} = [letter_all{nb_sub} letter];
    s_row_all{nb_sub,nb_file} = s_row_iter;
    s_col_all{nb_sub,nb_file} = s_col_iter;
    
    s_row_target = [s_row_target s_row_iter(r(nb_sub,nb_file) , :)];
     s_col_target = [s_col_target s_col_iter(c(nb_sub,nb_file) , :)];
    index = true(1, 6);
    index(r(nb_sub,nb_file)) = false;
    s_row_nontarget = [s_row_nontarget reshape(s_row_iter(index,:),[],75)];
    %%%%%%%%%%%%%%%%%%
    if Adaptive_flag == 1 
    if  adaptation < max_adaptation    
        adaptation = adaptation + 1;
        x_new = x(1:iter*12,:); codet = code(1:iter*12);
        y_new = -1*ones(size(x_new,1),1);
        y_new(codet==row) = 1; y_new(codet==col+6) = 1;
        tic
         for k = 1:length(classifiers)
             classifiers(k) = svmincrement(x_new, y_new, classifiers(k));
         end
        learning_time(nb_sub, nb_file) = toc;
    end
    end     
         %%%%%%%%%%%%%
         
    end
    perf(nb_sub) = sum(letter_all{nb_sub}==correct_result{nb_sub})/length(files_dir);
    
    clear filet 

    
end

    for nb_sub = 1:length(subject)
    mean_iter_all(nb_sub)  = mean(iter_all{nb_sub});
    end
    
    fprintf('\n');
    g=sprintf('%.2f ', perf*100);
    fprintf('%s\n ', g)
    fprintf('Avg Acc = %f\n ', mean(perf*100))
    
    g=sprintf('%.2f  ', mean_iter_all);
    fprintf('%s\n  ', g)
    fprintf('Avg iter = %f\n ', mean(mean_iter_all))
    
    %% ANALYSIS on SCORES
    
    s_row_all_accumulated = cell(length(subject), 1);
    s_col_all_accumulated = s_row_all_accumulated;
    margin_1_row = cell(length(subject), 1); %subtraction between max and second max of 6 rows
    margin_1_col = margin_1_row; %subtraction between max and second max of 6 cols
     
for nb_sub = 1:length(correct_result)
    for nb_file = 1:length(correct_result{nb_sub})
        for nb_iter = 1:15
              s_row_all_accumulated{nb_sub, nb_file}(:,nb_iter) = mean(s_row_all{nb_sub, nb_file}(:,1:nb_iter), 2);
              s_col_all_accumulated{nb_sub, nb_file}(:,nb_iter) = mean(s_col_all{nb_sub, nb_file}(:,1:nb_iter), 2);
              
         margin_1_row{nb_sub, nb_file}(nb_iter) = s_row_all_accumulated{nb_sub, nb_file}(r(nb_sub, nb_file),nb_iter) ...
        - max(s_row_all_accumulated{nb_sub, nb_file}(s_row_all_accumulated{nb_sub, nb_file}(:,nb_iter) ~=  s_row_all_accumulated{nb_sub, nb_file}(r(nb_sub, nb_file),nb_iter),nb_iter));
              
         margin_1_col{nb_sub, nb_file}(nb_iter) = max(s_col_all_accumulated{nb_sub, nb_file}(:,nb_iter)) ...
        - max(s_col_all_accumulated{nb_sub, nb_file}(s_col_all_accumulated{nb_sub, nb_file}(:,nb_iter)<max(s_col_all_accumulated{nb_sub, nb_file}(:,nb_iter)),nb_iter));    
    
        end
    end
end

avg_margin_1 =  zeros(length(subject),max_iter);
for nb_sub = 1:length(subject)
    for nb_iter = 1:max_iter
        for nb_file = 1:length(correct_result{nb_sub})
        avg_margin_1(nb_sub,nb_iter) = avg_margin_1(nb_sub,nb_iter) + margin_1_row{nb_sub, nb_file}(nb_iter);
        end 
    end
    
    for nb_iter = 1:max_iter
        avg_margin_1(nb_sub,nb_iter) = avg_margin_1(nb_sub,nb_iter)/ length(correct_result{i});
    end
    
end
    
    fprintf('Max Learning Time = %f\n ', mean(max(learning_time')))  

    
    
    
    