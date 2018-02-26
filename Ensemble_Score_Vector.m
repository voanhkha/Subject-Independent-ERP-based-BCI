%% SUBJECT-INDEPENDENT ERP-BASED BRAIN-COMPUTER INTERFACE
% Ensemble_Score_Vector routine
% Author: Kha Vo, voanhkha@yahoo.com.
% Inputs: 
% classifiers = a struct of multiple classifiers
% x = vector for generating scores
% Outputs: 
% y_individual = a matrix of scores of each classifier on each element in x
% y_sum = sum of scores of all classifiers on x (sum of y_individual)

%% BEGIN
function [y_sum, y_individual]=Ensemble_Score_Vector(classifiers,x)

span=1;
nbclassifier=length(classifiers);
y_individual = zeros(size(x,1),nbclassifier);
y_sum = 0;   

for ii=1:nbclassifier 
        xt=x;
%         xsup=classifiers(ii).xsup;
%         w=classifiers(ii).w;
%         b=classifiers(ii).b;
        mnormalize=classifiers(ii).mnormalize;
        stdnormalize=classifiers(ii).stdnormalize;
        [xt]=normalise(xt,mnormalize,stdnormalize);
     
%         if ~isfield(classifiers,'kernel')  || ~isfield(classifiers,'kerneloption')
%             kernel='poly';
%             kerneloption=1;
%         else
%             kernel=classifiers(ii).kernel;
%             kerneloption=classifiers(ii).kerneloption;
%         end;
        
        %y=svmval(xt,xsup,w,b,kernel,kerneloption,span);
        y=svmscore(xt',classifiers(ii));
        y_sum = y_sum + y; 
        y_individual(:,ii) = y;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end;
end
%% END
