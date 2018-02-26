% SVMEVAL - Evaluates a support vector machine at the given data points.
%
% Syntax: [f,K] = svmscore(X,a,b,ind,X_mer,y_mer,type,scale)
%         (evaluates the given SVM at the data points contained in X)
%
%         [f,K] = svmscore(X)
%         (evaluates the SVM in memory at the data points contained in X)
%
%      f: SVM output for the evaluation vectors
%      K: kernel matrix containing dot products in feature space between
%         the margin and error vectors (rows of K) and the column vectors in X
%         (columns of K)
%      X: matrix of evaluation vectors stored columnwise
%      a: alpha coefficients
%      b: bias
%    ind: cell array containing indices of margin, error and reserve vectors
%         ind{1}: indices of margin vectors
%         ind{2}: indices of error vectors
%         ind{3}: indices of reserve vectors
%  X_mer: matrix of margin, error and reserve vectors stored columnwise
%  y_mer: column vector of class labels (-1/+1) for margin, error and reserve vectors
%   type: kernel type
%           1: linear kernel        K(x,y) = x'*y
%         2-4: polynomial kernel    K(x,y) = (scale*x'*y + 1)^type
%           5: Gaussian kernel with variance 1/(2*scale)
%  scale: kernel scale
%
% Version 3.22e -- Comments to diehl@alumni.cmu.edu
%

function [f,K] = svmscore(X_eval,classifier)

% flags for example state
MARGIN    = 1;
ERROR     = 2;
RESERVE   = 3;
UNLEARNED = 4;
  global a b ind X Y type scale
  if (nargin == 2)
      setClassifier(classifier);
  end
   % define arguments
%    a     = classifier.a;
%    b     = classifier.b;
%    ind   = classifier.ind;
%    X     = classifier.X;
%    Y     = classifier.Y;
%   mnormalise = classifier.mnormalise;
 %  stdnormalise = classifier.stdnormalise;
 %  type  = 1; % inner product
 %  scale = 1; % not important

   %X_eval = normalise(X_eval', mnormalise, stdnormalise);
   %X_eval = X_eval';

% evaluate the SVM

% find all of the nonzero coefficients
% (note: when performing kernel perturbation, ind{MARGIN} and ind{ERROR}
%  do not necessarily identify all of the nonzero coefficients)
indu = a(ind{UNLEARNED}) > 0;
indu = ind{UNLEARNED}(indu);
indr = a(ind{RESERVE}) > 0;
indr = ind{RESERVE}(indr);
indme = [ind{MARGIN} ind{ERROR}];

K = [];
f = b;
if (~isempty(indme))
   K = kernel(X(:,indme),X_eval,type,scale);
   f = f + K'*(Y(indme).*a(indme));
end;
if (~isempty(indu))
   f = f + kernel(X(:,indu),X_eval,type,scale)'*(Y(indu).*a(indu));
end;
if (~isempty(indr))
   f = f + kernel(X(:,indr),X_eval,type,scale)'*(Y(indr).*a(indr));
end;

