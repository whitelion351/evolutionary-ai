from evolutionai import AgentManager, Agent


if __name__ == '__main__':
    manager = AgentManager(population=100, pool_size=0.1, network_layout=[24, 16, 16, 4], env_name="BipedalWalkerHardcore-v3",
                           continuous_action=True, episodes_per_agent=10, generations=500, use_watchdog=300, watchdog_penalty=100, 
                           obs_scale=1, action_scale=1, is_client=True,
                           n_procs=4)
