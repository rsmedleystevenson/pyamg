cd '~/Dropbox/Multigrid/pyamg_fork/pyamg/dev/';

% Test 1
% A = [1,0,3,1,-1;
%      2,2,0,1,-2;
%      1,1,0,0,0];
% b = [-1;1;0];
% Ct = [1, 1;
%       1, 2;
%       1, 3;
%       1, 4;
%       1, 5];
% d = [1,-1];

% Test 2
% A = [ 1, 1,0, 0,2;
%      -1, 3,1, 0,0;
%       2, 1,0,-1,0;
%      -2,-2,4, 0,0;
%       3, 0,1, 3,3];
% b = [-2;-1;0;1;2];
% Ct = [-1;
%       1;
%       -1;
%       1;
%       -1];
% d = [2];
  
% Test 3
% A = [ 1, 2,0, 0,2;
%       1, 2,1, 0,0;
%       0, 0,0,-1,0;
%       0, 0,4, 0,0;
%       0, 0,1, 3,3];
% b = [-2;-1;0;1;2];
% Ct = [-1;
%       1;
%       -1;
%       1;
%       -1];
% d = [2];

A = [ 0,-1; 0, -1; -1, 0; -1, 0];
b = [0;0;1;0];
Ct = [2,2];
d = [1];



[m,n] = size(A);
s = length(d);

[Qc,Rc] = qr(Ct);


S = A*Qc;

temp_vec = Rc(1:s,1:s)' \ d;
b2 = b - S(:,1:s)*temp_vec;

[Qs,Rs] = qr(S(:,(s+1):end));
b2 = Qs'*b2;
temp = min(size(Rs));
sol = Rs(1:temp,1:temp) \ b2(1:temp);

x = [temp_vec; sol];
x = Qc*x;


%% Solve min ||xA - b|| s.t. xC = d

A = [ 0, -1, -1; -1, 0, 0];
b = [0;1;0];
C = [2; 2];
d = [1];



[m,n] = size(A);
s = length(d);

[Qc,Rc] = qr(C);


S = A'*Qc;

temp_vec = Rc(1:s,1:s)' \ d;
b2 = b - S(:,1:s)*temp_vec;

[Qs,Rs] = qr(S(:,(s+1):end));
b2 = Qs'*b2;
temp = min(size(Rs));
sol = Rs(1:temp,1:temp) \ b2(1:temp);

x = [temp_vec; sol];
x = Qc*x;



