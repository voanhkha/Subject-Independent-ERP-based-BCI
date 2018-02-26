%% SUBJECT-INDEPENDENT ERP-BASED BRAIN-COMPUTER INTERFACE
% setClassifier routine
% Author: Kha Vo, voanhkha@yahoo.com.
% Syntax: classifier = getClassifier
% Usage: get all the global variables related to classifier's parameters
% and return to 'classifier'


function setClassifier(classifier)

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

    a = classifier.a;
    b = classifier.b;
    C = classifier.C;
   % classifier.deps=deps;
    g = classifier.g;
    ind = classifier.ind;
    %classifier.max_reserve_vectors=max_reserve_vectors;
    Q = classifier.Q;
    Rs = classifier.Rs;
   % scale = classifier.scale;
   % type = classifier.type;
   % classifier.uind=uind;
    X = classifier.X;
    Y = classifier.Y;
    %classifier.mnormalize=mnormalize;
    %classifier.stdnormalize=stdnormalize;
    %classifier.filename=file;

end
