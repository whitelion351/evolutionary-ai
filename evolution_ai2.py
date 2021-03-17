from copy import deepcopy
from math import tanh
from random import randrange, random, choice, shuffle
import gym
# import pybullet_envs
import numpy as np


class AgentManager:
    def __init__(self, population=50, network_layout=None, env_name=None, continuous_action=True, generations=999,
                 episodes_per_agent=1, pool_size=10, obs_scale=1, action_scale=1, input_nodes=2, output_nodes=2,
                 timestep_limit=None):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.generations = generations
        self.obs_scale_factor = obs_scale
        self.action_scale_factor = action_scale
        self.total_population = population
        self.env_name = env_name
        self.continuous_action = continuous_action
        self.timestep_limit = timestep_limit
        self.network_layout = network_layout
        self.episodes_per_agent = episodes_per_agent
        self.mutate_chance = 0.2
        self.max_mutate_amount = 0.2
        self.population_to_keep = pool_size

        if self.env_name is None:
            self.train_env = None
            self.play_env = None
        else:
            self.set_env(env_name)

        self.agents = self.create_population()
        self.scores = [0 for _ in range(self.total_population)]

    def set_env(self, env_name):
        print("setting env to", env_name)
        self.env_name = env_name
        self.train_env = gym.make(self.env_name)
        self.play_env = gym.make(self.env_name)
        print("environment details")
        print(self.train_env.observation_space)
        print(self.train_env.action_space)

    def create_population(self):
        agents = []
        for a in range(self.total_population):
            agent = Agent(agent_id=a, continuous_action=self.continuous_action,
                          input_nodes=self.input_nodes, output_nodes=self.output_nodes)
            agents.append(agent)
        return agents

    def train(self):
        print("begin training...")
        self.scores = [0 for _ in range(self.total_population)]
        best_agents = []
        for e in range(self.generations):
            for p in range(self.total_population):
                agent_score = 0
                print(f"\rplaying agent {p}", end="")
                for ep in range(self.episodes_per_agent):
                    do_render = True if len(best_agents) > 0 and p in best_agents[:3] and ep == 0 else False
                    agent_score += self.play_episode(self.agents[p], self.train_env, do_render=do_render)
                agent_score /= self.episodes_per_agent
                # print(f"epoch {e} agent {p} score {agent_score}")
                self.scores[p] = agent_score
                self.agents[p].score = agent_score
            # print(f"epoch {e} finished")
            best_agents = self.process_agents(e)
            # best_score = self.play_episode(self.agents[best_agents[0]], self.train_env)
            # print("best agent score =", best_score)
            self.scores = [0 for _ in range(self.total_population)]
            for a in self.agents:
                a.reset()

    def play_episode(self, agent, env, do_render=False, timestep_limit=None):
        agent_score = 0
        timestep_limit = timestep_limit if timestep_limit is not None else self.timestep_limit
        obs_scale_factor = 1 if self.obs_scale_factor is None else self.obs_scale_factor
        action_scale_factor = 1 if self.action_scale_factor is None else self.action_scale_factor
        obs = list(np.array(env.reset()) / obs_scale_factor)
        timestep = 0
        done = False
        if do_render:
            env.render()
        while not done:
            # if abs(max(obs)) > 1:
            #     obs_scale_factor += 0.5
            #     print(max(obs), "increasing scale_factor to", obs_scale_factor)
            action = agent.predict(obs)
            action = list(np.array(action) * action_scale_factor) if self.continuous_action is True else action
            obs, reward, done, info = env.step(action)
            obs = list(np.array(obs) / obs_scale_factor)
            if do_render:
                env.render()
            agent_score += reward
            timestep += 1
            if timestep_limit is not None and timestep >= timestep_limit:
                done = True
        return agent_score

    def process_agents(self, epoch):
        avg_score = 0
        for score_item in self.scores:
            avg_score += score_item
        avg_score /= len(self.scores)
        overall_avg_score = avg_score
        best_agents = []
        self.scores.sort(reverse=True)
        high_score = self.scores[0]
        for i in range(self.population_to_keep):
            shuffled_list = list(range(len(self.agents)))
            shuffle(shuffled_list)
            for agent_i in shuffled_list:
                _agent = self.agents[agent_i]
                if _agent.score == self.scores[i] and _agent.agent_id not in best_agents:
                    best_agents.append(_agent.agent_id)
                    break
        print(f"\rgen: {epoch + 1} avg: {round(overall_avg_score, ndigits=2)} high: {round(high_score, ndigits=2)} "
              f"top 3: {best_agents[:3]} best layout: {self.agents[best_agents[0]].network_layout}")
        for p in range(self.total_population):
            if p not in best_agents:
                self.agents[p].network = self.cross_over(best_agents)
                self.agents[p].network = self.mutate_network([p])
        return best_agents

    def cross_over(self, agent_list):
        net_a = deepcopy(self.agents[choice(agent_list)].network)
        net_b = self.agents[choice(agent_list)].network

        for layer_index, layer in enumerate(net_a):
            for node_index, node in enumerate(layer):
                for weight_index, weight in enumerate(node):
                    if random() >= 0.5:
                        net_a[layer_index][node_index][weight_index] = net_b[layer_index][node_index][weight_index]
        return net_a

    @staticmethod
    def random_float():
        return (random() - 0.5) * 2

    def mutate_network(self, agent_list):
        chosen_one = choice(agent_list)
        chosen_network = deepcopy(self.agents[chosen_one].network)
        for layer in range(len(chosen_network)):
            for node in range(len(chosen_network[layer])):
                for weight in range(len(chosen_network[layer][node])):
                    chance = random()
                    if chance < self.mutate_chance:
                        _weight = chosen_network[layer][node][weight] + (self.random_float() * self.max_mutate_amount)
                        if abs(_weight) > 1:
                            _weight = self.random_float()
                        chosen_network[layer][node][weight] = _weight
        return chosen_network


class Agent:
    def __init__(self, agent_id=-1, network_layout=None, continuous_action=True, input_nodes=None, output_nodes=None):
        if agent_id < 0:
            print("creating agent with agent_id", agent_id)
        self.agent_id = agent_id
        self.score = 0
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.continuous_action = continuous_action
        self.min_layers = 2
        self.max_layers = 2
        self.min_layer_nodes = 16
        self.max_layer_nodes = 16
        self.network_layout = network_layout
        self.network = self.initialize_network()

    def initialize_network(self):
        if self.network_layout is None:
            new_layout = [self.input_nodes]
            new_layers = int(randrange(self.min_layers, self.max_layers + 1))
            for _new_layer in range(new_layers):
                new_layout.append(int(randrange(self.min_layer_nodes, self.max_layer_nodes + 1)))
            new_layout.append(self.output_nodes)
            self.network_layout = new_layout

        finished_network = []
        for layer in range(len(self.network_layout) - 1):
            layer_nodes = []
            for _node in range(self.network_layout[layer]):
                weights_per_node = []
                for weight in range(self.network_layout[layer + 1]):
                    weights_per_node.append((random() - 0.5) * 2)
                layer_nodes.append(weights_per_node)
            finished_network.append(layer_nodes)
        return deepcopy(finished_network)

    def predict(self, obs):
        return self.calculate_action(obs)

    def calculate_action(self, input_layer):
        weights = self.network
        network_activations = [input_layer]
        for layer in range(len(self.network_layout) - 1):
            result = []
            interim_result = []
            for inputs in range(len(input_layer)):
                for weight in range(len(weights[layer][inputs])):
                    y = input_layer[inputs] * weights[layer][inputs][weight]
                    result.append(y)
                interim_result.append(result)
                result = []

            input_layer = []
            for index in range(len(interim_result[0])):
                activation = 0
                for from_node in range(len(interim_result)):
                    activation += interim_result[from_node][index]
                input_layer.append(tanh(activation))
            network_activations.append(input_layer)
        if self.continuous_action:
            return input_layer
        return np.argmax(input_layer)

    def reset(self):
        self.score = 0


if __name__ == "__main__":
    manager = AgentManager(population=50, pool_size=5, env_name="LunarLanderContinuous-v2", generations=500,
                           input_nodes=8, output_nodes=2, continuous_action=True, episodes_per_agent=10,
                           obs_scale=1, action_scale=1)
    manager.train()
