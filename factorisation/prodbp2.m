function [Afin,Sfin, trt,tat,B,tatref] = prodb2

N = 64;
M = 128;
L = 2000;
rho = 0.1;
lambda = 0.1;
beta = 5000;

D = randn(N, M);
D = D./(repmat(sqrt(sum(D.^2)),[N 1]));

B = D'*D;
noise = 0.001;

Z = randn(M,L).*( rand(M,L) < rho );
X = D * Z;
lambda = lambda * mean((sum(X.^2)));

[A, S, ~]= svd(B,0);
A = A';

K = max(S(:));
XX = pinv(D)*X;
Z1 = sign(XX).*max(0, abs(XX) - lambda/K);
disti = sqrt(mean((Z(:)-Z1(:)).^2));
Znoise = Z + disti*randn(size(Z));

tatref = lambda*( sum(max(0,sum(abs(A*Z)) - sum(abs(Z)) - sum(abs(A*Znoise)) + sum(abs(Znoise)))))
Rref = norm(B)*eye(M) - B;
tatref2 =real( .5*trace((Z-Z1)'*Rref*(Z-Z1)))


if 0 | tatref < tatref2
    noise = 0.0001;
    RR = noise*randn(M);
    %R = .5*(R+R');
    %[V,LL]=eig(R);
    %R = V * max(0,LL) * V';
    %RR=0*B;
    R = RR'*RR;
    [A, S, ~]= svd(B+R,0);
    A = A';
    fprintf('init from svd \n')
else
    A= eye(M);
    S = norm(B)*eye(M);
    R = A'*S*A - B;
    [V, LL]=eig(R);
    RR =  max(0,LL.^(1/2)) * V';
    %RR = chol(R);
    fprintf('init from identity \n')
end


niters =1000;
lr = 1e-5;

mom=0;
momS=0;
momRR=0;

rhomom=0.5;
rhomomS=0.5;
rhomomRR = 0.5;


for n=1:niters
    if mod(n,niters/4)==niters/4 -1
       lr = lr * 0.7; 
    end
        
%    Znoise = Z + disti*randn(size(Z));
    R = A'*S*A - B;
    dA = commbp(A, Z, Znoise, lambda);
    dR = tracebp(R, Z-Z1);
    resid = (R - RR'*RR);
    dR = dR + beta*resid;
    dRR = -2*RR* beta*resid; 
    
    %keyboard
    
    dS = diag(diag(A * dR * A'));
    dA = dA + 2* S * A * dR;
    dA = dA - A*(dA')*A;

    trt(n) = real(trace(.5* (Z-Z1)'*R*(Z-Z1)));
    tat(n) = lambda*( sum(max(0,sum(abs(A*Z)) - sum(abs(Z)) - sum(abs(A*Znoise)) + sum(abs(Znoise)))));
    tet(n) = beta*.5*norm(resid(:));
    err(n) =  trt(n) + tat(n) + tet(n);   
 
    mom = rhomom * mom - lr * dA;
    momS = rhomomS * momS - lr * dS;
    momRR = rhomomRR * momRR - lr * dRR;
    A = A + mom;
    [U2, S2, V2]=svd(A,0);
    A = U2*V2';

    S = S + momS;
    RR = RR + momRR;

    if 0 | mod(n,10)==1
        err(n)
        %hold off; plot(trt,'r');hold on; plot(tat,'b');plot(tet,'g');
        plot(eig(R))
        drawnow
        %keyboard
    end
    
end
Afin = A;
Sfin = S;

err(end)
tatref
tatref2

%do a step and evaluate the loss 
Z2 = A'*softhresh( (A - pinv(S)*A*B)*Z1 + pinv(S)*A*D'*X, lambda*repmat(diag(S).^(-1),1,size(Z1,2)));
Z2b = softhresh( Z2 - (1/norm(B))*B*Z2 + 1/norm(B)*D'*X, (lambda/norm(B))*ones(size(Z2)));

Z3 = softhresh( Z1 - (1/norm(B))*B*Z1 + 1/norm(B)*D'*X, (lambda/norm(B))*ones(size(Z2)));
Z3b = softhresh( Z3 - (1/norm(B))*B*Z3 + 1/norm(B)*D'*X, (lambda/norm(B))*ones(size(Z2)));

norm(Z2b(:))
norm(Z2b(:)-Z3b(:))

rec2 = X - D*Z2;
errore2 = .5*norm(rec2(:))^2 + lambda * sum(abs(Z2(:)))

rec3 = X - D*Z3;
errore3 = .5*norm(rec3(:))^2 + lambda * sum(abs(Z3(:)))


rec2 = X - D*Z2b;
errore2b = .5*norm(rec2(:))^2 + lambda * sum(abs(Z2b(:)))

rec3 = X - D*Z3b;
errore3b = .5*norm(rec3(:))^2 + lambda * sum(abs(Z3b(:)))


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




