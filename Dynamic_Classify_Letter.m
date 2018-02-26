%% SUBJECT-INDEPENDENT ERP-BASED BRAIN-COMPUTER INTERFACE
% Dynamic_Classify_Letter routine
% Author: Kha Vo, voanhkha@yahoo.com.

function [iter_stop, classified_letter, row, col, s_row_iter, s_col_iter] = ...
    Dynamic_Classify_Letter_2(x, stimuli_code, classifiers, typedataset, theta_parameters)

theta_1_tilde = theta_parameters(1);
theta_2_tilde = theta_parameters(2);

switch typedataset
    case 'BCI2'
        matrix=['ABCDEF';'GHIJKL';'MNOPQR';'STUVWX';'YZ1234';'56789_']';
        max_iter = 15;
    case 'BCI3'
        matrix=['ABCDEF';'GHIJKL';'MNOPQR';'STUVWX';'YZ1234';'56789_']';
           max_iter = 15;
    case 'Akimpech'
        matrix=['ABCDEF';'GHIJKL';'MNOPQR';'STUVWX';'YZ1234';'56789_'];
           max_iter = 15;
    case 'ALS'
        matrix=['ABCDEF';'GHIJKL';'MNOPQR';'STUVWX';'YZ1234';'56789_'];
           max_iter = 10;
end

iter = 0;
s_row = zeros(6,1); s_col = zeros(6,1);
s_row_iter = zeros(6, max_iter); s_col_iter = zeros(6, max_iter); 
s_row_indi= zeros(6, length(classifiers)); s_col_indi = s_row_indi;
stopflag_1 = 0; stopflag_2 = 0;


while stopflag_1==0 || stopflag_2==0
   M = zeros(6,6);
   iter = iter+1;
   xt = x((iter-1)*12+1:iter*12,:);
   code_t = stimuli_code((iter-1)*12+1 : iter*12);
   
      [yt, yt_individual]= Ensemble_Score_Vector(classifiers,xt);
   
    for index = 1:6
        s_row(index) = s_row(index) + yt(code_t==index);
        s_row_iter(index,iter) = yt(code_t==index);
        
        s_col(index) = s_col(index) + yt(code_t==index+6);
        s_col_iter(index,iter) = yt(code_t==index+6);
        
        s_row_indi(index,:) = s_row_indi(index,:) +  yt_individual(code_t==index,:);
        s_col_indi(index,:) = s_col_indi(index,:) +  yt_individual(code_t==index+6,:);
    end
    
    % Dynamic Criteria Checking
    if stopflag_1 == 0 % only perform condition checking if stopflag_1 = 0
    for p = 1:length(classifiers)
        [~,r] = max(s_row_indi(:,p));
        [~,c] = max(s_col_indi(:,p));
        M(r,c) = M(r,c) + 1;
    end
    
    denomi_theta_1 = (1/6)*(sum(s_row-min(s_row)) + sum(s_col-min(s_col))); %denominator of theta_1
    nomi_theta_1 = max(s_row) - max(s_row(s_row<max(s_row)))...
            + max(s_col) - max(s_col(s_col<max(s_col))); %nominator of theta_1
    theta_1 = nomi_theta_1 / denomi_theta_1;
    [theta_2, maxindex] = max(M(:));
    theta_2 = theta_2 / length(classifiers);
          
    [~, r1] = max(s_row); 
    [~, c1] = max(s_col);
    [r2, c2] = ind2sub(size(M),maxindex);
    
          if (theta_1 >= theta_1_tilde && theta_2 >= theta_2_tilde && r1==r2 && c1==c2)
            stopflag_1 = 1;
            iter_stop = iter;
          end
    
    end
    % End of Dynamic Criteria Checking
    
    if iter == max_iter
        stopflag_2 = 1;
        if stopflag_1 == 0;
            iter_stop = max_iter;
            stopflag_1 = 1;
        end 
    end
    

   
end

classified_letter = matrix(r1, c1);
row = r1; col = c1;

end