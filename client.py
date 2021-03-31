from evolutionai import AgentManager


if __name__ == '__main__':
    manager = AgentManager(population=100, pool_size=0.1, network_layout=[8, 16, 16, 4], env_name="LunarLander-v2",
                           continuous_action=False, episodes_per_agent=3, use_watchdog=500, watchdog_penalty=100,
                           generations=500, obs_scale=1, action_scale=1, n_procs=2, is_client=True)
