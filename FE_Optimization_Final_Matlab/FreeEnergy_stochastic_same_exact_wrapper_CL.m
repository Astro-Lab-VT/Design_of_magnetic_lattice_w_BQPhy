function std = FreeEnergy_stochastic_same_exact_wrapper_CL(opt_space,x,y)
    [pop,~]=size(opt_space);
    std=zeros(pop,1);
    for i=1:pop
        std(i)=FreeEnergy_stochastic_same_exact_CL(opt_space(i,:),x,y);
    end
end