function  [x_out,m,sigma]=normalise(x_in,m,sigma)
%  USAGE: 
% [x_out, m, sigma] = normalise(x_in): normalises x_in internally and
%  returns mean 'm' and standard deviation 'sigma'. Often used in training.
% [x_out] = normalise(x_in, m, sigma): normalise x_in using the input mean
%  and deviation. Often used in testing a new vector.

%  Author : KHA VO. 
%  Updated: March 2017.

if (nargin == 1)
    m = mean(x_in);
    sigma = std(x_in);
end
    lengthx =size(x_in,1);
    x_out=(x_in - ones(lengthx,1)*m)./(ones(lengthx,1)*sigma);
end