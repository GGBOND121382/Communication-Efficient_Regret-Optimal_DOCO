def determine_comm_const_AGD(O_comm_budget, comm_budget, lwr_const, upp_const, time_list, num_of_clients, mu,
                             eps=1e-10):
    lwr_comm_cost = O_comm_budget(lwr_const, time_list, num_of_clients, mu)
    upp_comm_cost = O_comm_budget(upp_const, time_list, num_of_clients, mu)
    # assert lwr_comm_cost < comm_budget
    # assert upp_comm_cost > comm_budget
    if (lwr_comm_cost > comm_budget or upp_comm_cost < comm_budget):
        return None
    while (abs(upp_const - lwr_const) > eps):
        mid_const = (lwr_const + upp_const) / 2
        mid_comm_cost = O_comm_budget(mid_const, time_list, num_of_clients, mu)
        if (mid_comm_cost < comm_budget):
            lwr_const = mid_const
        elif (mid_comm_cost > comm_budget):
            upp_const = mid_const
        else:
            return mid_const
    return lwr_const



# comm_const, time_horizon, num_of_clients, mu, num_parallel, dimension, Lip_cons, C_norm, M_cons

def determine_comm_const_CP(O_comm_budget, comm_budget, lwr_const, upp_const, time_horizon, num_of_clients, mu,
                            num_parallel, dimension, Lip_cons, C_norm, M_cons, eps=1e-10):
    lwr_comm_cost = O_comm_budget(lwr_const, time_horizon, num_of_clients, mu, num_parallel, dimension, Lip_cons,
                                  C_norm, M_cons)
    upp_comm_cost = O_comm_budget(upp_const, time_horizon, num_of_clients, mu, num_parallel, dimension, Lip_cons,
                                  C_norm, M_cons)
    # assert lwr_comm_cost < comm_budget
    # assert upp_comm_cost > comm_budget
    if(lwr_comm_cost > comm_budget or upp_comm_cost < comm_budget):
        return None
    while (abs(upp_const - lwr_const) > eps):
        mid_const = (lwr_const + upp_const) / 2
        mid_comm_cost = O_comm_budget(mid_const, time_horizon, num_of_clients, mu, num_parallel, dimension, Lip_cons,
                                  C_norm, M_cons)
        if (mid_comm_cost < comm_budget):
            lwr_const = mid_const
        elif (mid_comm_cost > comm_budget):
            upp_const = mid_const
        else:
            return mid_const
    return lwr_const
