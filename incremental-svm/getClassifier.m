
% Syntax: classifier = getClassifier
% Usage: get all the global variables related to classifier's parameters
% and return to 'classifier'
% Author: Kha Vo.
%

function classifier = getClassifier

% define global variables 
global a;                     % alpha coefficients
global b;                     % bias
global C;                     % regularization parameters
%global deps;                  % jitter factor in kernel matrix
global g;                     % partial derivatives of cost function w.r.t. alpha coefficients
global ind;                   % cell array containing indices of margin, error, reserve and unlearned vectors
%global max_reserve_vectors;   % maximum number of reserve vectors stored
global Q;                     % extended kernel matrix for all vectors
global Rs;                    % inverse of extended kernel matrix for margin vectors   
global scale;                 % kernel scale
global type;                  % kernel type
%global uind;                  % user-defined example indices
global X;                     % matrix of margin, error, reserve and unlearned vectors stored columnwise
global Y;                     % column vector of class labels (-1/+1) for margin, error, reserve and unlearned vectors

    classifier.a=a;
    classifier.b=b;
    classifier.C=C;
   % classifier.deps=deps;
    classifier.g=g;
    classifier.ind=ind;
    %classifier.max_reserve_vectors=max_reserve_vectors;
    classifier.Q=Q;
    classifier.Rs=Rs;
    classifier.scale=scale;
    classifier.type=type;
   % classifier.uind=uind;
    classifier.X=X;
    classifier.Y=Y;
    %classifier.mnormalize=mnormalize;
    %classifier.stdnormalize=stdnormalize;
    %classifier.filename=file;

end
