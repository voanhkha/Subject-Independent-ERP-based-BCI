% LEARN - Increments the specified example into the current SVM solution.  
%         Assumes alpha_c = 0 initially.
%
% Syntax: nstatus = learn(indc,rflag)
%
% nstatus: new status for indc
%    indc: index of the example to learn
%   rflag: flag indicating whether or not to check if any reserve vectors
%          become margin vectors during learning
%
% Version 3.22e -- Comments to diehl@alumni.cmu.edu
%

function nstatus = learn(indc,rflag)

% flags for example state
MARGIN    = 1;
ERROR     = 2;
RESERVE   = 3;
UNLEARNED = 4;

% define global variables 
global a b C deps g ind perturbations Q Rs scale type X Y  % alpha coefficients

% compute g(indc) 
[f_c,K] = svmscore(X(:,indc));
g(indc) = Y(indc)*f_c - 1;

% if g(indc) > 0, place this example into the reserve set directly
if (g(indc) >= 0)
   
   % move the example to the reserve set
   %param.a = a; param.C = C; param.ind = ind;
   bookkeeping(indc,UNLEARNED,RESERVE);
   nstatus = RESERVE;
   
   return;
end;

% compute Qcc and Qc if necessary
num_MVs = length(ind{MARGIN});
Qc = cell(3,1);
if (num_MVs == 0)
   if (~isempty(ind{ERROR}))
   	Qc{ERROR} = (Y(ind{ERROR})*Y(indc)).*kernel(X(:,ind{ERROR}),X(:,indc),type,scale);
   end;
else
	Qc{MARGIN} = (Y(ind{MARGIN})*Y(indc)).*K(1:num_MVs);
	if (~isempty(ind{ERROR}))
   	Qc{ERROR} = (Y(ind{ERROR})*Y(indc)).*K(num_MVs+1:length(K));
	end;
end;
if (~isempty(ind{RESERVE}))
   Qc{RESERVE} = (Y(ind{RESERVE})*Y(indc)).*kernel(X(:,ind{RESERVE}),X(:,indc),type,scale);
end;
Qcc = kernel(X(:,indc),X(:,indc),type,scale) + deps;

converged = 0;
while (~converged)
   
   perturbations = perturbations + 1;
   
   if (num_MVs > 0)  % change in alpha_c permitted
   
      % compute Qc, beta and gamma
      beta = -Rs*[Y(indc) ; Qc{MARGIN}];
      gamma = zeros(size(Q,2),1);
      ind_temp = [ind{ERROR} ind{RESERVE} indc];
      gamma(ind_temp) = [Qc{ERROR} ; Qc{RESERVE} ; Qcc] + Q(:,ind_temp)'*beta;
      
      % check if gamma_c < 0 (kernel matrix is not positive semi-definite)
      if (gamma(indc) < 0)
         error('LEARN: gamma_c < 0');
      end;
      
   else  % change in alpha_c not permitted since the constraint on the sum of the
         % alphas must be preserved.  only b can change.  
      
      % set beta and gamma
      beta = Y(indc);
      gamma = Y(indc)*Y;
      
   end;
   
   % minimum acceptable parameter change (change in alpha_c (num_MVs > 0) or b (num_MVs = 0))
   % param.a = a; param.C = C; param.g = g; param.ind = ind;
   [min_delta_param,indss,cstatus,nstatus] = min_delta_acb(indc,gamma,beta,1,rflag);
   
   % update a, b, and g
   if (num_MVs > 0)
      a(indc) = a(indc) + min_delta_param;
      a(ind{MARGIN}) = a(ind{MARGIN}) + beta(2:num_MVs+1)*min_delta_param;
   end;   
   b = b + beta(1)*min_delta_param;
   g = g + gamma*min_delta_param;
         
   % update Qc and perform bookkeeping         
   converged = (indss == indc); % ONLY STOP WHEN indc is learned, or indss==indc
   if (converged)
      cstatus = UNLEARNED;
     	Qc{nstatus} = [Qc{nstatus} ; Qcc];
  	else
  		ind_temp = find(ind{cstatus} == indss);
  		Qc{nstatus} = [Qc{nstatus} ; Qc{cstatus}(ind_temp)];
  		Qc{cstatus}(ind_temp) = [];
   end;
%    param.a = a; param.C = C; param.ind = ind;
   [indco,removed_i] = bookkeeping(indss,cstatus,nstatus);
   if ((nstatus == RESERVE) & (removed_i > 0)) %#ok<AND2>
      Qc{nstatus}(removed_i) = [];
   end;
      
   % set g(ind{MARGIN}) to zero
   g(ind{MARGIN}) = 0;
   
   % update Rs and Q if necessary
   if (nstatus == MARGIN)
              
      num_MVs = num_MVs + 1;
      if (num_MVs > 1)
         if (converged)
            gamma = gamma(indss);
         else
               
            % compute beta and gamma for indss            
            beta = -Rs*Q(:,indss);
            gamma = kernel(X(:,indss),X(:,indss),type,scale) + deps + Q(:,indss)'*beta;
            
         end;
      end;
            
      % expand Rs and Q
      updateRQ(beta,gamma,indss);
      
   elseif (cstatus == MARGIN)      
              
      % compress Rs and Q      
      num_MVs = num_MVs - 1;
      updateRQ(indco);
            
   end;         
   
end;
