% SVMTRAIN - Trains a support vector machine incrementally
%            using the L1 soft margin approach developed by
%            Cauwenberghs for two-class problems.
%
% Syntax: [a,b,g,ind,uind,X_mer,y_mer,Rs,Q] = svmtrain(X,y,C,type,scale)
%         [a,b,g,ind,uind,X_mer,y_mer,Rs,Q] = svmtrain(X,y,C,type,scale,uind)
%         (trains a new SVM on the given examples)
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

%function [a,b,g,ind,uind,X,y,Rs,Q] = svmtrain_diehl(X_new,y_new,C_new,varargin)
function model = svm_newmodel(X_new,y_new,C_new,mnormalize, stdnormalize)
% flags for example state
MARGIN    = 1;
ERROR     = 2;
RESERVE   = 3;
UNLEARNED = 4;
C_new = C_new*ones(size(y_new));    

%define global variables 
global a b C deps g ind max_reserve_vectors Q Rs X Y type scale
% initialize variables
   deps = 1e-3; 
   max_reserve_vectors = 3000;    
   num_examples = size(X_new,2);       
   type = 1; scale = 1;
   a = zeros(num_examples,1);          
   b = 0;                              
   C = C_new;                          
   g = zeros(num_examples,1);
   ind = cell(4,1);
   ind{UNLEARNED} = 1:num_examples;
 %  kernel_evals = 0;
 %  perturbations = 0;
   Q = y_new';
   Rs = Inf;
 %  uind = uind_new;
  %[X_new,mnormalise,stdnormalise]=normalise(X_new);
   X = X_new;                          
   Y = y_new;
 
% begin incremental learning - enforce all constraints on each iteration
num_learned = 1;
% disp('Beginning training.');
while (any(ind{UNLEARNED}))
   
   % randomly select example
   i = round(rand*(length(ind{UNLEARNED})-1)) + 1;
   indc = ind{UNLEARNED}(i);
%  indc = ind{UNLEARNED}(1);

	% learn example
   learn(indc,1);
   
   if (mod(num_learned,50) == 0)
      s = sprintf('Learned %d examples.',num_learned);
     % disp(s);
   end;
   num_learned = num_learned + 1;
   
end;
if (mod(num_learned-1,50) ~= 0)
   s = sprintf('Learned %d examples.',num_learned-1);
  % disp(s);
end;
% disp('Training complete!');

% remove all but the closest reserve vectors from the dataset if necessary
if (length(ind{RESERVE}) == max_reserve_vectors)
   ind_keep = [ind{MARGIN} ind{ERROR} ind{RESERVE}];
   a = a(ind_keep);
   g = g(ind_keep);
   Q = Q(:,ind_keep);   
%   uind = uind(ind_keep);
   X = X(:,ind_keep);
   Y = Y(ind_keep);
   ind{MARGIN} = 1:length(ind{MARGIN});
   ind{ERROR} = length(ind{MARGIN}) + (1:length(ind{ERROR}));
   ind{RESERVE} = length(ind{MARGIN}) + length(ind{ERROR}) + (1:length(ind{RESERVE}));
end;

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
model.mnormalize = mnormalize;
model.stdnormalize = stdnormalize;

% summary statistics
% s = sprintf('\nMargin vectors:\t\t%d',length(ind{MARGIN}));
% disp(s);
% s = sprintf('Error vectors:\t\t%d',length(ind{ERROR}));
% disp(s);
% s = sprintf('Reserve vectors:\t%d',length(ind{RESERVE}));
% disp(s);
% s = sprintf('Kernel evaluations:\t%d\n',kevals);
% disp(s);
