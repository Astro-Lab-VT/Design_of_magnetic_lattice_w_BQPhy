function FlipedGOmega = FlipedGOmegaFn_APseq_CL(M_org, N_org)
    num_omega = max(M_org, N_org) + 1;

    sigma = 0.25;
    difference = 4 / (num_omega - 1);

    GaussianSigma = zeros(1, num_omega - 1);
    for n = 1:num_omega-1
        GaussianSigma(n) = 4 - n * difference;
    end

    GaussianSigma = GaussianSigma(1:end-1);

    GaussianOmega = sigma * GaussianSigma;
    FlipedGOmega = sort(GaussianOmega, 'descend');

    FlipedGOmega = FlipedGOmega / sum(FlipedGOmega);
    FlipedGOmega = FlipedGOmega(:);
end