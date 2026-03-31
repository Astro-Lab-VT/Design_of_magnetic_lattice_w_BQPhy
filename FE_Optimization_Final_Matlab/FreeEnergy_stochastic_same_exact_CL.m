function Free_energy_stddeviation = FreeEnergy_stochastic_same_exact_CL(x_spin, x, y)
% FreeEnergy_stochastic_same_exact_CL
% Exact-output cached version.
% Caches all geometry/topology work that does not
% depend on x_spin.
%
% Inputs:
%   x_spin : vectorized spin configuration
%   x      : external field samples
%   y      : thermal energy samples (kBT)
%
% Output:
%   Free_energy_stddeviation : standard deviation of Free_energy over samples

    x_spin = x_spin > 0;
    n = sqrt(numel(x_spin));
    if abs(n - round(n)) > eps
        error('x_spin must form a square lattice');
    end
    n = round(n);

    % -------------------------------------------------
    % Persistent cache
    % -------------------------------------------------
    persistent CACHE
    if isempty(CACHE) || ~isfield(CACHE, 'n') || CACHE.n ~= n
        CACHE = build_free_energy_cache(n);
    end

    % -------------------------------------------------
    % Spin mapping
    % -------------------------------------------------
    mainmatrix = x_spin + 1;
    mainmatrix(mainmatrix > 1) = -1;
    mainmatrix = reshape(mainmatrix, n, n);

    sigVec = mainmatrix(:);
    Nsites = CACHE.Numel_m_size;

    % -------------------------------------------------
    % Per-site energy without field
    % -------------------------------------------------
    E0 = zeros(Nsites, 1);
    for nID = 1:Nsites
        idx = CACHE.neighbor_idx{nID};
        w   = CACHE.neighbor_w{nID};
        si  = sigVec(nID);

        E0(nID) = sum(-(si .* sigVec(idx) .* w));
    end

    % -------------------------------------------------
    % Free energy over samples
    % -------------------------------------------------
    x = x(:);
    y = y(:);

    beta = 1 ./ y;                       % Nsamp x 1
    Henergy = E0 - sigVec * x.';         % Nsites x Nsamp
    RHS = exp(-Henergy .* beta.');       % Nsites x Nsamp
    Free_energy = (-(log(sum(RHS, 1))) ./ beta.') ./ Nsites;

    Free_energy_stddeviation = std(Free_energy(:), 1); % <-- population std? wait
end




