function Omegamatrix = OmegamatrixFn_CL(orgmainmatrix, FlipedGOmega, M_org, N_org, rowi_sigma_i, colj_sigma_i)
% OmegamatrixFn
% Builds an omega-weighted matrix centered at a selected lattice location.
%
% Inputs:
%   orgmainmatrix : original matrix used as the base template
%   FlipedGOmega  : descending omega weights
%   M_org         : number of rows in the lattice
%   N_org         : number of columns in the lattice
%   rowi_sigma_i  : row index of the center site
%   colj_sigma_i  : column index of the center site
%
% Output:
%   Omegamatrix   : matrix with row/column weights assigned according to
%                   distance from the selected center site
%
% Notes:
%   - Rows below and above the center are assigned weights from
%     FlipedGOmega in increasing distance order.
%   - Columns left and right of the center are assigned weights in the same way.
%   - The center entry is set to 1.

i = rowi_sigma_i;
j = colj_sigma_i;

mainmatrix = orgmainmatrix;
M = M_org;
N = N_org;

%% Indices around the center
ivec = i + 1 : M;      % bottom side
jvec = i - 1 : -1 : 1; % top side
kvec = j - 1 : -1 : 1; % left side
lvec = j + 1 : N;      % right side

%% Pad vectors to equal length
maxel = max([numel(ivec), numel(jvec), numel(kvec), numel(lvec)]);

if numel(ivec) < maxel
    ivec(numel(ivec)+1:maxel) = 0;
end

if numel(jvec) < maxel
    jvec(numel(jvec)+1:maxel) = 0;
end

if numel(kvec) < maxel
    kvec(numel(kvec)+1:maxel) = 0;
end

if numel(lvec) < maxel
    lvec(numel(lvec)+1:maxel) = 0;
end

%% Assign omega weights by distance from the center
for x = 1:maxel
    if ivec(x) > 0
        mainmatrix(ivec(x), :) = FlipedGOmega(x);
    end

    if jvec(x) > 0
        mainmatrix(jvec(x), :) = FlipedGOmega(x);
    end

    if kvec(x) > 0
        mainmatrix(:, kvec(x)) = FlipedGOmega(x);
    end

    if lvec(x) > 0
        mainmatrix(:, lvec(x)) = FlipedGOmega(x);
    end
end

%% Restore center value
mainmatrix(rowi_sigma_i, colj_sigma_i) = 1;
Omegamatrix = mainmatrix;

end