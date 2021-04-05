from evolutionai import AgentManager


if __name__ == '__main__':
    manager = AgentManager(population=100, pool_size=0.1, network_layout=[8, 16, 16, 2], env_name="LunarLanderContinuous-v2",
                           continuous_action=True, episodes_per_agent=10, generations=500, obs_scale=1, action_scale=1,
                           n_procs=4, is_client=True)
