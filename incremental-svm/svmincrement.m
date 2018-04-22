% SVMTRAIN - Trains a support vector machine incrementally
%            using the L1 soft margin approach developed by
%            Cauwenberghs for two-class problems.
%
% Syntax: [a,b,g,ind,uind,X_mer,y_mer,Rs,Q] = svmtrain(X,y,C,type,scale)
%         [a,b,g,ind,uind,X_mer,y_mer,Rs,Q] = svmtrain(X,y,C,type,scale,uind)
%         (trains a new SVM on the given examples)
%
%         [a,b,g,ind,uind,X_mer,y_mer,Rs,Q] = svmtrain(X,y,C)
%         [a,b,g,ind,uind,X_mer,y_mar,Rs,Q] = svmtrain(X,y,C,uind)
%         (trains the current SVM in memory on the given examples)
%
%      a: alpha coefficients
%      b: bias
%      g: partial derivatives of cost function w.r.t. alpha coefficients
%    ind: cell array containing indices of margin, error and reserve vectors
%         ind{1}: indices of margin vectors
%         ind{2}: indices of error vectors
%         ind{3}: indices of reserve vectors
%   uind: column vector of user-defined example indices (used for unlearning specified examples)
%  X_mer: matrix of margin, error and reserve vectors stored columnwise
%  y_mer: column vector of class labels (-1/+1) for margin, error and reserve vectors
%     Rs: inverse of extended kernel matrix for margin vectors
%      Q: extended kernel matrix for all vectors
%      X: matrix of training vectors stored columnwise
%      y: column vector of class labels (-1/+1) for training vectors
%      C: soft-margin regularization parameter(s)
%         dimensionality of C       assumption
%         1-dimensional vector      universal regularization parameter
%         2-dimensional vector      class-conditional regularization parameters (-1/+1)
%         n-dimensional vector      regularization parameter per example
%         (where n = # of examples)
%   type: kernel type
%           1: linear kernel        K(x,y) = x'*y
%         2-4: polynomial kernel    K(x,y) = (scale*x'*y + 1)^type
%           5: Gaussian kernel with variance 1/(2*scale)
%  scale: kernel scale
%
% Version 3.22e -- Comments to diehl@alumni.cmu.edu
%

%function [a,b,g,ind,uind,X,y,Rs,Q] = svmincrement(X_new,y_new,C_new,old_classifier)
function new_classifier = svmincrement(X_new,y_new,old_classifier)
% flags for example state
MARGIN    = 1;
ERROR     = 2;
RESERVE   = 3;
UNLEARNED = 4;
%define global variables 
global a;                     % alpha coefficients
global b;                     % bias
global C;                     % regularization parameters 
global deps;                  % jitter factor in kernel matrix
global g;                     % partial derivatives of cost function w.r.t. alpha coefficients
global ind;                   % cell array containing indices of margin, error, reserve and unlearned vectors
% global kernel_evals;          % kernel evaluations
global max_reserve_vectors;   % maximum number of reserve vectors stored
% global perturbations;         % number of perturbations
global Q;                     % extended kernel matrix for all vectors
global Rs;                    % inverse of extended kernel matrix for margin vectors   
global scale;                 % kernel scale
global type;                  % kernel type
global uind;                  % user-defined example indices
global X;                     % matrix of margin, error, reserve and unlearned vectors stored columnwise
global Y;

 %     uind_new = zeros(size(y_new));
  
a = old_classifier.a;                     % alpha coefficients
b = old_classifier.b;                     % bias
C =  old_classifier.C;                     % regularization parameters                 % jitter factor in kernel matrix
g =  old_classifier.g;                     % partial derivatives of cost function w.r.t. alpha coefficients
ind = old_classifier.ind;                   % cell array containing indices of margin, error, reserve and unlearned vectors
Q = old_classifier.Q;                     % extended kernel matrix for all vectors
Rs = old_classifier.Rs;                    % inverse of extended kernel matrix for margin vectors   
X = old_classifier.X;                     % matrix of margin, error, reserve and unlearned vectors stored columnwise
Y = old_classifier.Y;                     % column vector of class labels (-1/+1) for margin, error, reserve and unlearned vectors
mnormalize = old_classifier.mnormalize;
stdnormalize = old_classifier.stdnormalize;

X_new = normalise(X_new,mnormalize, stdnormalize);
X_new = X_new';

% initialize variables
 deps = 1e-3;
 max_reserve_vectors = 3000;    

   num_examples = size(X,2);
   num_new_examples = size(X_new,2);
   
   a = [a ; zeros(num_new_examples,1)];
   C = [C ; C(1)*ones(size(y_new))];
   g = [g ; zeros(num_new_examples,1)];
   ind{UNLEARNED} = (1:num_new_examples) + num_examples;
   
   % assumes currently that there are no duplicate examples in the data - may not necessarily be true!
   Q_new = [y_new' ; (Y(ind{MARGIN})*y_new').*kernel(X(:,ind{MARGIN}),X_new,type,scale)];
   
   Q = [Q Q_new];  
   %uind = [uind ; uind_new];
   X = [X X_new];
   Y = [Y ; y_new];
   
   % num_examples = num_examples + num_new_examples;
   
% begin incremental learning - enforce all constraints on each iteration
num_learned = 1;
%disp('Beginning training.');
while (any(ind{UNLEARNED}))
   
   % randomly select example
   i = round(rand*(length(ind{UNLEARNED})-1)) + 1;
   indc = ind{UNLEARNED}(i);
%  indc = ind{UNLEARNED}(1);

	% learn example
   % current_model = getClassifier;
   learn(indc,1);
   
   if (mod(num_learned,50) == 0)
 %     s = sprintf('Learned %d examples.',num_learned);
 %     disp(s);
   end;
   num_learned = num_learned + 1;
   
end;
if (mod(num_learned-1,50) ~= 0)
   s = sprintf('Learned %d examples.',num_learned-1);
%   disp(s);
end;
%disp('Training complete!');

% remove all but the closest reserve vectors from the dataset if necessary
if (length(ind{RESERVE}) == max_reserve_vectors)
   ind_keep = [ind{MARGIN} ind{ERROR} ind{RESERVE}];
   a = a(ind_keep);
   g = g(ind_keep);
   Q = Q(:,ind_keep);   
   uind = uind(ind_keep);
   X = X(:,ind_keep);
   Y = Y(ind_keep);
   ind{MARGIN} = 1:length(ind{MARGIN});
   ind{ERROR} = length(ind{MARGIN}) + (1:length(ind{ERROR}));
   ind{RESERVE} = length(ind{MARGIN}) + length(ind{ERROR}) + (1:length(ind{RESERVE}));
end;

% summary statistics
% s = sprintf('\nMargin vectors:\t\t%d',length(ind{MARGIN}));
% disp(s);
% s = sprintf('Error vectors:\t\t%d',length(ind{ERROR}));
% disp(s);
% s = sprintf('Reserve vectors:\t%d',length(ind{RESERVE}));
% disp(s);
% s = sprintf('Kernel evaluations:\t%d\n',kevals);
% disp(s);

model.a = a;
model.b = b;
model.C = C;
model.g = g;
model.ind = ind;
model.X = X;
model.Y = Y;
model.Rs = Rs;
model.Q = Q;
% model.scale = scale;
% model.type = type;
model.mnormalize = old_classifier.mnormalize;
model.stdnormalize = old_classifier.stdnormalize;
new_classifier = model;
