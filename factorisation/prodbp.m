function [Afin,Sfin, trt,tat,tet,B,tatref] = prodbp

N = 64;
M = 128;
L = 2000;
rho = 0.1;
lambda = 0.5;

D = randn(N, M).^3;
D = D./(repmat(sqrt(sum(D.^2)),[N 1]));

B = D'*D;
noise = 0.001;

Z = randn(M,L).*( rand(M,L) < rho );
X = D * Z;
lambda = lambda * mean((sum(X.^2)));


[A, S, ~]= svd(B,0);
A = A';

Sinv = (diag(S)+eps).^(-1);
K = max(S(:));
XX = pinv(D)*X;
Z1 = sign(XX).*max(0, abs(XX) - lambda/K);
disti = sqrt(mean((Z(:)-Z1(:)).^2));
Znoise = Z + disti*randn(size(Z));

tatref = lambda*( sum(max(0,sum(abs(A*Z)) - sum(abs(Z)) - sum(abs(A*Znoise)) + sum(abs(Znoise)))))
Rref = norm(B)*eye(M) - B;
%Rrefbis = diag(sum(abs(B))) - B ;
%Rrefbis = diag((sum(B.^2)).^(1/2)) - B;
tatref2 =real( .5*trace((Z-Z1)'*Rref*(Z-Z1)))
%tatref2bis =real( .5*trace((Z-Z1)'*Rrefbis*(Z-Z1)))
%norm(Rref)
%norm(Rrefbis)

%keyboard

if tatref < tatref2

    YY = randn(size(B));
    YY = YY*YY';
    YY = YY/sqrt(size(B,1));
[A, S, ~]= svd(B,0);
A = A';
%S = (norm(B)/(norm(A)^2))*eye(M);
fprintf('init from svd \n')
else
   
    A= eye(M);
    S = norm(B)*eye(M);
    fprintf('init from identity \n')
end


niters =2000;
kappa = 2500;
nu = 2500;
lr = 1e-5;
mom=0;
momS=0;
rhomom=0.5;
rhomomS=0.5;



for n=1:niters
    if mod(n,niters/4)==niters/4 -1
       lr = lr * 0.9; 
    end
    % F = A * B * (A');
    %St = diag( sum(abs(F)));
    Rt = A' * S * A - B;
    %sanity check
    %plot(eig(Rt),'.');
    %keyboard
    Znoise = Z + disti*randn(size(Z));
    
    dA = commbp(A, Z, Znoise, lambda);
    dR = tracebp(Rt, Z-Z1);

    %add a term that penalizes non-psd
    % - nu * min(0,y^T R y)
    %Zdumb = randn(size(Z));
    [vv,dd]=eig(Rt);
    II=find(diag(dd)<0);
    ddneg = diag(dd);
    ddneg = ddneg(II);
    Zdumb = vv(:,II);%*diag(ddneg);
    %nothing = repmat((diag(Zdumb'*Rt*Zdumb) < 0),1,size(Z,1))';
    %size(nothing)
    %size(Z)
    dRbis = -nu* Zdumb * Zdumb';
    %(Z.*nothing) * ((Z').*(nothing'));
    %norm(dRbis(:)) 
    %norm(dR(:))
    dR = dR + dRbis;
    
    [tmp, dS]=resbp(A, S, dR);
    dA = dA + tmp;
    %dF = diagbp(F, dS);
    %tmp = prod0bp(A, B, dF);
    %dA = dA + tmp;
    
    %add a term that penalizes non-unitarity
    if 1

        dA = dA - A*(dA')*A;
        
    %project into space of unitary matrices
    tutu=0;
    else
   
    % kappa || A A^T - I ||_F^2
    tutu = A*A' - eye(M);
    dA = dA + kappa*tutu * A;
    end
    
    trt(n) = real(trace(.5* (Z-Z1)'*Rt*(Z-Z1)));
    tat(n) = lambda*( sum(max(0,sum(abs(A*Z)) - sum(abs(Z)) - sum(abs(A*Znoise)) + sum(abs(Znoise)))));
    tet(n) = kappa*sum(tutu(:).^2);
    err(n) =  trt(n) + tat(n) + tet(n);   
    
    mom = rhomom * mom - lr * dA;
    A = A + mom;
    momS = rhomomS * momS - lr * dS;
    S = S + momS;

    if 1
    [U2, S2, V2]=svd(A,0);
    A = U2*V2';
    end
    % A = A - lr * dA;
   
    
    
    %norm(A(:))
    %trt(n)
    %tat(n)
    if mod(n,50)==1
            %mean(nothing(:))
    err(n)
    plot(eig(Rt))
    drawnow
    end
    
end
Afin = A;
Sfin = S;

err(end)
tatref
tatref2

%do a step and evaluate the loss 
Z2 = A'*softhresh( (A - pinv(S)*A*B)*Z1 + pinv(S)*A*D'*X, lambda*repmat(diag(S).^(-1),1,size(Z1,2)));
Z2b = softhresh( Z2 - (1/norm(D))*B*Z2 + 1/norm(D)*D'*X, (lambda/norm(B))*ones(size(Z2)));

Z3 = softhresh( Z1 - (1/norm(D))*B*Z1 + 1/norm(D)*D'*X, (lambda/norm(B))*ones(size(Z2)));
Z3b = softhresh( Z3 - (1/norm(D))*B*Z3 + 1/norm(D)*D'*X, (lambda/norm(B))*ones(size(Z2)));

norm(Z2b(:))
norm(Z2b(:)-Z3b(:))

rec2 = X - D*Z2b;
errore2 = .5*norm(rec2(:))^2 + lambda * sum(abs(Z2b(:)))

rec3 = X - D*Z3b;
errore3 = .5*norm(rec3(:))^2 + lambda * sum(abs(Z3b(:)))


end


function out = softhresh(in, t)

out = sign(in).*max(0, abs(in) - t);

end

function dA = prod0bp(A, B, din)

%A is (m x k)
%B is (k x k)
%out is (m x m)
%din is (m x m)

%out = A * B * A';

dA = 2 * din * A * B;

end

function dF = diagbp( F, din)

tmp = diag(din);
dF = diag(tmp)*sign(F);

end

function [dA, dS] = resbp(A, S, din)

dA = 2 * din * A' * S;
dS = A * din * A';
dS = diag(diag(dS));

end

function dR = tracebp( R , Z) 

dR = Z * Z';

end


function dcom = commbp(A, Z, Z1, lambda)

t0 = sum( abs(A*Z) - abs(Z) - abs(A*Z1) + abs(Z1));
I0 = repmat( (t0 > 0), size(Z,1), 1);

t1 = (I0.*sign(A*Z))*Z';
t2 = (I0.*sign(A*Z1))*Z1';
dcom = lambda*(t1 - t2);

end




