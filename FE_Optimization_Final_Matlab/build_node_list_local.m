function node_list = build_node_list_local(main_nodes, neighbor_nodes)
    main_nodes = main_nodes(:);
    num_main = numel(main_nodes);
    num_neighbors = size(neighbor_nodes, 2);

    node_list = zeros(num_main * num_neighbors, 2);
    q = 1;

    for p = 1:num_neighbors:(num_main * num_neighbors)
        node_list(p:p + num_neighbors - 1, 1) = main_nodes(q);
        node_list(p:p + num_neighbors - 1, 2) = reshape(neighbor_nodes(q, :), [], 1);
        q = q + 1;
    end
end