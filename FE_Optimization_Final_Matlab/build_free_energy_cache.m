function cache = build_free_energy_cache(n)

    M_org = n;
    N_org = n;
    Numel_m_size = M_org * N_org;
    m = reshape(1:Numel_m_size, M_org, N_org);
    m_size = n;

    %% Center neighbors
    num_center = (M_org - 2) * (N_org - 2);
    mle_node = zeros(num_center, 8);
    k1 = 0;
    for i = 2:M_org-1
        for j = 2:N_org-1
            k1 = k1 + 1;
            e1 = m(i,   j+1);   % right
            e2 = m(i+1, j);     % bottom
            e3 = m(i-1, j);     % top
            e4 = m(i-1, j-1);   % top-left
            e5 = m(i,   j-1);   % left
            e6 = m(i+1, j+1);   % bottom-right
            e7 = m(i-1, j+1);   % top-right
            e8 = m(i+1, j-1);   % bottom-left
            mle_node(k1, :) = [e1 e2 e3 e4 e5 e6 e7 e8];
        end
    end

    %% Left column neighbors
    lc_node = zeros(M_org - 2, 5);
    k2 = 0;
    for i = 2:M_org-1
        k2 = k2 + 1;
        le1 = m(i,   2);   % right
        le2 = m(i+1, 1);   % bottom
        le3 = m(i+1, 2);   % bottom-right
        le4 = m(i-1, 1);   % top
        le5 = m(i-1, 2);   % top-right
        lc_node(k2, :) = [le1 le2 le3 le4 le5];
    end

    %% Right column neighbors
    rc_node = zeros(M_org - 2, 5);
    k3 = 0;
    for i = 2:M_org-1
        k3 = k3 + 1;
        re1 = m(i,   N_org-1); % left
        re2 = m(i+1, N_org);   % bottom
        re3 = m(i-1, N_org);   % top
        re4 = m(i-1, N_org-1); % top-left
        re5 = m(i+1, N_org-1); % bottom-left
        rc_node(k3, :) = [re1 re2 re3 re4 re5];
    end

    %% Bottom row neighbors
    br_node = zeros(N_org - 2, 5);
    k4 = 0;
    for j = 2:N_org-1
        k4 = k4 + 1;
        be1 = m(M_org,   j+1); % right
        be2 = m(M_org,   j-1); % left
        be3 = m(M_org-1, j);   % top
        be4 = m(M_org-1, j-1); % top-left
        be5 = m(M_org-1, j+1); % top-right
        br_node(k4, :) = [be1 be2 be3 be4 be5];
    end

    %% Top row neighbors
    tr_node = zeros(N_org - 2, 5);
    k5 = 0;
    for j = 2:N_org-1
        k5 = k5 + 1;
        te1 = m(1, j-1);   % left
        te2 = m(2, j);     % bottom
        te3 = m(1, j+1);   % right
        te4 = m(2, j+1);   % right-bottom
        te5 = m(2, j-1);   % left-bottom
        tr_node(k5, :) = [te1 te2 te3 te4 te5];
    end

    %% Corners
    cleftop     = [m(1,2),           m(2,1),           m(2,2)];
    cleftbottom = [m(M_org-1,1),     m(M_org,2),       m(M_org-1,2)];
    crhttop     = [m(1,N_org-1),     m(2,N_org),       m(2,N_org-1)];
    crhtbottom  = [m(M_org-1,N_org), m(M_org,N_org-1), m(M_org-1,N_org-1)];

    node_corner = transpose([ ...
        1,              1,              1, ...
        m(M_org,1),     m(M_org,1),     m(M_org,1), ...
        m(1,N_org),     m(1,N_org),     m(1,N_org), ...
        m(M_org,N_org), m(M_org,N_org), m(M_org,N_org); ...
        cleftop,        cleftbottom,    crhttop,       crhtbottom]);

    %% Node-center list
    mcenter = transpose(m(2:M_org-1, 2:N_org-1));
    mcenter = reshape(mcenter, [1, numel(mcenter)]);
    node_center = build_node_list_local(mcenter, mle_node);

    %% Edge lists
    m_lc = m(2:M_org-1, 1);
    node_lc = build_node_list_local(m_lc(:), lc_node);

    m_rc = m(2:M_org-1, N_org);
    node_rc = build_node_list_local(m_rc(:), rc_node);

    m_br = m(M_org, 2:N_org-1);
    node_br = build_node_list_local(m_br(:), br_node);

    m_tr = m(1, 2:N_org-1);
    node_tr = build_node_list_local(m_tr(:), tr_node);

    %% Final edge list in exact concatenation order
    node_f = [node_corner; node_tr; node_br; node_rc; node_lc; node_center];

    %% Exact de-dup logic
    [M_nf, N_nf] = size(node_f);
    for l = 1:M_nf
        r = node_f(l, 1);
        c = node_f(l, 2);
        dx = find(node_f(:,1) == c);

        for ll = 1:numel(dx)
            if node_f(dx(ll)) == c
                [row, column] = ind2sub([M_nf, N_nf], dx(ll));
                if column < N_nf && node_f(row, column+1) == r
                    node_f(row, column)   = 0;
                    node_f(row, column+1) = 0;
                end
            end
        end
    end

    node_f(~any(node_f, 2), :) = [];
    node_f(:, ~any(node_f, 1)) = [];

    %% Graph and neighborhoods
    G = graph(node_f(:,1), node_f(:,2));
    FlipedGOmega = FlipedGOmegaFn_APseq_CL(M_org, N_org);
    wsnum = Numel_m_size - 1;

    nodeIDs = cell(Numel_m_size, 1);
    for nID = 1:Numel_m_size
        nodeIDs{nID} = nearest(G, nID, wsnum);
    end

    %% Precompute row/col
    [r_of, c_of] = ind2sub([M_org, N_org], 1:Numel_m_size);

    %% Precompute exact weights implied by OmegamatrixFn_CL overwrite logic
    neighbor_idx = cell(Numel_m_size, 1);
    neighbor_w   = cell(Numel_m_size, 1);

    for nID = 1:Numel_m_size
        ri = r_of(nID);
        cj = c_of(nID);

        neigh = nodeIDs{nID}(:);
        rr = r_of(neigh);
        cc = c_of(neigh);

        % Exact overwrite behavior:
        % Omega(idx) = FlipedGOmega(max(abs(dr), abs(dc))) for non-center
        cheb_dist = max(abs(rr - ri), abs(cc - cj));

        w = FlipedGOmega(cheb_dist);

        neighbor_idx{nID} = neigh;
        neighbor_w{nID} = w(:);
    end

    cache.n = n;
    cache.M_org = M_org;
    cache.N_org = N_org;
    cache.Numel_m_size = Numel_m_size;
    cache.m_size = m_size;
    cache.neighbor_idx = neighbor_idx;
    cache.neighbor_w = neighbor_w;
end