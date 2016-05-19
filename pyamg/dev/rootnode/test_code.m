cd '/Users/ben/Dropbox/Multigrid/RootNode';

% Simple example matrix with 4 course points and 12 fine points. 
% Aggregates are constructed s.t. root nodes are not connected to other
% aggregates. 
Afc = [ 1 0 0 0;
        0 1 0 0;
        1 0 0 0;
        0 0 1 0;
        1 0 0 0;
        0 0 1 0;
        0 0 0 1;
        0 0 0 1;
        0 1 0 0;
        0 1 0 0;
        0 0 1 0; 
        0 0 0 1];

% Sort A_fc s.t. all elements in a given aggregate are adjacent rows.
Afc = sortrows(Afc);

% Take transpose for A_cf by symmetry
Acf = Afc';

% Form product needed for NII and pseudoinverse
prod = Afc*Acf;
prod_inv = pinv(prod);




%%

G = [17,0,0,12,3;
    0,9,4,0,1;
    0,4,0,0,7;
    12,0,0,17,9;
    3,1,7,9,0];

sparsity = ceil(G / 100);

chi1 = sparsity(1,:);
chi1 = diag(chi1);

e1 = zeros(1,length(G));
e1(1) = 1;

chi1*G*chi1



