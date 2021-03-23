from copy import deepcopy
from math import tanh
from random import randrange, random, choice, shuffle
from time import time
from threading import Thread
from multiprocessing import Process
from time import sleep
import gym
import numpy as np
import tkinter as tk
import pickle
import os
import psutil


class AgentManager:
    def __init__(self, population=50, network_layout=None, env_name=None, continuous_action=True, generations=999,
                 episodes_per_agent=1, pool_size=0.25, obs_scale=1, action_scale=1, input_nodes=2, output_nodes=2,
                 timestep_limit=None, use_watchdog=-1, watchdog_penalty=0, n_procs=-1):
        self.n_procs = psutil.cpu_count() if n_procs == -1 else n_procs
        self.watchdog_penalty = watchdog_penalty
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.generations = generations
        self.obs_scale_factor = obs_scale
        self.action_scale_factor = action_scale
        self.total_population = population
        self.env_name = env_name
        self.continuous_action = continuous_action
        self.timestep_limit = timestep_limit
        self.use_watchdog = use_watchdog
        self.network_layout = network_layout
        self.episodes_per_agent = episodes_per_agent
        self.mutate_chance = 0.1
        self.max_mutate_amount = 0.2
        self.population_to_keep = int(self.total_population * pool_size)
        print(f"pool/population size {self.population_to_keep}/{self.total_population}")
        self.model_dir = "models/"
        self.worker_dir = "workers/"
        self.render_done = False
        self.render_watchdog = False
        self.render_best = False
        self.render_all_agents = False
        self.render_all_episodes = False
        for w in range(self.n_procs):
            self.remove_file(self.worker_dir+"worker"+str(w))
            self.remove_file(self.worker_dir+"worker"+str(w)+"result")

        if self.env_name is None:
            self.train_env = None
            self.play_env = None
        else:
            self.set_env(env_name)

        self.agents = self.create_population()
        self.scores = [0 for _ in range(self.total_population)]

        print(f"starting {self.n_procs} processes")
        for w in range(self.n_procs):
            proc = Process(name="worker"+str(w), target=self.worker, daemon=True, args=(w,))
            proc.start()

        sleep(1.0)
        print("starting frontend")
        self.frontend = MainWindow(self)
        self.frontend.mainloop()

    @staticmethod
    def remove_file(file_to_remove):
        try:
            os.remove(file_to_remove)
        except OSError:
            pass

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
            agent = Agent(agent_id=a, network_layout=self.network_layout, continuous_action=self.continuous_action,
                          input_nodes=self.input_nodes, output_nodes=self.output_nodes)
            agents.append(agent)
        return agents

    def worker(self, w_id):
        env = gym.make(self.env_name)
        proc = psutil.Process()
        proc.cpu_affinity()  # arg would be cpu to run on
        print("started worker", w_id, "with affinity", proc.cpu_affinity())
        while True:
            try:
                worker_work = pickle.load(open(self.worker_dir+"worker" + str(w_id), "rb"))
                agent = worker_work[0]
                eps_per_agent = worker_work[1]["eps"]
                agent_score = 0
                for ep in range(eps_per_agent):
                    do_render = worker_work[1]["do_render"]
                    do_render = False if ep != 0 and worker_work[1]["all_eps"] is False else do_render
                    agent_score += self.play_episode(agent, env, do_render=do_render,
                                                     render_watchdog=worker_work[1]["render_watchdog"],
                                                     render_done=worker_work[1]["render_done"],
                                                     timestep_limit=self.timestep_limit)
                agent_score /= self.episodes_per_agent
                self.remove_file(self.worker_dir+"worker" + str(w_id))
                pickle.dump(agent_score, open(self.worker_dir+"worker"+str(w_id)+"result", "wb"))
                sleep(0.1)
            except (FileNotFoundError, EOFError):
                sleep(0.1)

    def train(self):
        print("begin training...")
        self.scores = [0 for _ in range(self.total_population)]
        best_agents = []
        for e in range(self.generations):
            next_agent = 0
            completed_agents = 0
            proc_ids = [-1 for _ in range(self.n_procs)]
            while completed_agents < self.total_population:
                for proc_id in range(self.n_procs):
                    if next_agent < self.total_population and proc_ids[proc_id] == -1:
                        do_render = True if next_agent in best_agents[:3] and self.render_best \
                            or self.render_all_agents \
                            else False
                        pickle.dump([self.agents[next_agent],
                                     {"agent_id": next_agent, "eps": self.episodes_per_agent, "do_render": do_render,
                                      "render_watchdog": self.render_watchdog, "render_done": self.render_done,
                                      "all_eps": self.render_all_episodes}], open(self.worker_dir+"worker"+str(proc_id), "wb"))
                        proc_ids[proc_id] = next_agent
                        next_agent += 1
                        sleep(0.1)
                    else:
                        try:
                            agent_score = pickle.load(open(self.worker_dir+"worker"+str(proc_id)+"result", "rb"))
                            if agent_score is None:
                                self.remove_file(self.worker_dir+"worker"+str(proc_id)+"result")
                                pickle.dump([self.agents[proc_ids[proc_id]],
                                             {"agent_id": proc_ids[proc_id],
                                              "eps": self.episodes_per_agent, "do_render": False,
                                              "all_eps": self.render_all_episodes}],
                                            open(self.worker_dir+"worker" + str(proc_id), "wb"))
                            else:
                                self.scores[proc_ids[proc_id]] = agent_score
                                self.agents[proc_ids[proc_id]].score = agent_score
                                completed_agents += 1
                                proc_ids[proc_id] = -1
                                self.remove_file(self.worker_dir+"worker"+str(proc_id)+"result")
                        except (FileNotFoundError, EOFError):
                            sleep(0.1)
                a_status = f"gen: {e+1} agent: {next_agent-1}"
                self.frontend.control_window.status1_label_var.set(a_status)
            best_agents = self.process_agents(e)
            self.scores = [0 for _ in range(self.total_population)]
            for a in self.agents:
                a.reset()
        self.frontend.control_window.train_thread = None

    def play_episode(self, agent, env, do_render=False, render_watchdog=False, render_done=False, timestep_limit=None):
        agent_score = 0
        old_agent_score = 0
        timestep_limit = timestep_limit if timestep_limit is not None else self.timestep_limit
        obs_scale_factor = 1 if self.obs_scale_factor is None else self.obs_scale_factor
        action_scale_factor = 1 if self.action_scale_factor is None else self.action_scale_factor
        obs = list(np.array(env.reset()) / obs_scale_factor)
        timestep = 0
        done = False
        if do_render:
            env.render()
        while not done:
            if abs(max(obs)) > 1 and obs_scale_factor != 1:
                obs_scale_factor += 0.5
                print(max(obs), "increasing scale_factor to", obs_scale_factor)
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
            if self.use_watchdog != -1 and timestep != 0 and timestep % self.use_watchdog == 0:
                if agent_score <= old_agent_score:
                    # print(f"agent {agent.agent_id} died from watchdog")
                    done = True
                    agent_score -= self.watchdog_penalty
                old_agent_score = agent_score
                if not do_render and render_watchdog:
                    env.render()
        if not do_render and render_done:
            env.render()
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
        e_status = f"gen: {epoch + 1} avg: {round(overall_avg_score, ndigits=2)} high: {round(high_score, ndigits=2)}" \
                   f" top 3: {best_agents[:3]} best layout: {self.agents[best_agents[0]].network_layout}"
        print(e_status)
        self.frontend.control_window.status2_label_var.set(e_status)

        # all_agents = list(range(self.total_population))
        for p in range(self.total_population):
            if p not in best_agents:
                self.agents[p].network = self.cross_over(best_agents, best_agents)
                self.agents[p].network = self.mutate_network([p])
        return best_agents

    def cross_over(self, agent_list_a, agent_list_b):
        net_a = deepcopy(self.agents[choice(agent_list_a)].network)
        net_b = self.agents[choice(agent_list_b)].network

        for layer_index, layer in enumerate(net_a):
            for node_index, node in enumerate(layer):
                for weight_index, weight in enumerate(node):
                    if random() >= 0.5:
                        net_a[layer_index][node_index][weight_index] = net_b[layer_index][node_index][weight_index]
        return deepcopy(net_a)

    @staticmethod
    def random_float():
        return (random() - 0.5) * 2

    def mutate_network(self, agent_list):
        chosen_one = choice(agent_list)
        chosen_network = deepcopy(self.agents[chosen_one].network)
        for layer in range(len(chosen_network)):
            for node in range(len(chosen_network[layer])):
                for weight in range(len(chosen_network[layer][node])):
                    if random() < self.mutate_chance:
                        _weight = chosen_network[layer][node][weight] + (self.random_float() * self.max_mutate_amount)
                        if abs(_weight) > 1.0:
                            _weight = self.random_float()
                        chosen_network[layer][node][weight] = _weight
        return deepcopy(chosen_network)


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
                    weights_per_node.append((random() - 0.5))
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


class MainWindow(tk.Tk):
    def __init__(self, manager_object):
        super(MainWindow, self).__init__()
        self.title("Evolution AI Controller v1.0")
        self.font = ("helvetica", 10)
        self.update_delay = 0.5
        self.canvas = tk.Canvas(self, width=715, height=255, bg="#555555")
        self.canvas.pack()
        self.resizable(width=False, height=False)

        self.manager_object = manager_object
        self.control_window = ControlWindow(self)
        self.control_window.frame.place(x=10, y=10)


class ControlWindow:
    def __init__(self, root, width=700, height=240, bd=10, relief="ridge"):
        self.root = root
        self.font = ("helvetica", 10)
        self.frame = tk.Frame(root, width=width, height=height, bg="#AAAAAA", bd=bd, relief=relief)

        # The Start button (don't touch it)
        self.start_train_button = tk.Button(self.frame, width=5, bg="#CCCCCC", text="START", command=self.start_train)
        self.start_train_button.place(x=10, y=10)

        # Mutate controls
        self.mutate_chance_title = tk.Label(self.frame, width=11, bg="#AAAAAA", text="Mutate Chance")
        self.mutate_chance_title.place(x=10, y=52)
        self.chance_down_button = tk.Button(self.frame, width=1, bg="#CCCCCC", text="-", command=self.chance_down_func)
        self.chance_down_button.place(x=110, y=50)
        self.mutate_chance_label_var = tk.StringVar()
        self.mutate_chance_label_var.set(str(self.root.manager_object.mutate_chance))
        self.mutate_chance_label = tk.Label(self.frame, width=5, textvariable=self.mutate_chance_label_var)
        self.mutate_chance_label.place(x=128, y=52)
        self.chance_up_button = tk.Button(self.frame, width=1, bg="#CCCCCC", text="+", command=self.chance_up_func)
        self.chance_up_button.place(x=170, y=50)

        self.mutate_amount_title = tk.Label(self.frame, width=11, bg="#AAAAAA", text="Mutate Amount")
        self.mutate_amount_title.place(x=10, y=92)
        self.mutate_down_button = tk.Button(self.frame, width=1, bg="#CCCCCC", text="-", command=self.amount_down_func)
        self.mutate_down_button.place(x=110, y=90)
        self.mutate_amount_label_var = tk.StringVar()
        self.mutate_amount_label_var.set(str(self.root.manager_object.max_mutate_amount))
        self.mutate_amount_label = tk.Label(self.frame, width=5, textvariable=self.mutate_amount_label_var)
        self.mutate_amount_label.place(x=128, y=92)
        self.mutate_up_button = tk.Button(self.frame, width=1, bg="#CCCCCC", text="+", command=self.amount_up_func)
        self.mutate_up_button.place(x=170, y=90)

        # Save and Load buttons
        self.save_button = tk.Button(self.frame, width=5, text="SAVE", bg="#CCCCCC", command=self.save_pop)
        self.save_button.place(x=150, y=10)
        self.load_button = tk.Button(self.frame, width=5, text="LOAD", bg="#CCCCCC", command=self.load_pop)
        self.load_button.place(x=210, y=10)

        # Render options
        self.render_title = tk.Label(self.frame, width=11, bg="#AAAAAA", text="Render Options")
        self.render_title.place(x=440, y=12)
        self.render_done_checkbox = tk.Checkbutton(self.frame, width=5, height=1, bg="#AAAAAA",
                                                   text="Done", command=self.toggle_render_done)
        self.render_done_checkbox.place(x=310, y=35)
        self.render_watchdog_checkbox = tk.Checkbutton(self.frame, width=7, height=1, bg="#AAAAAA",
                                                       text="Watchdog", command=self.toggle_render_watchdog)
        self.render_watchdog_checkbox.place(x=375, y=35)
        self.render_best_checkbox = tk.Checkbutton(self.frame, width=3, height=1, bg="#AAAAAA",
                                                   text="Best 3", command=self.toggle_render_best)
        self.render_best_checkbox.place(x=465, y=35)
        self.render_all_agents_checkbox = tk.Checkbutton(self.frame, width=5, height=1, bg="#AAAAAA",
                                                         text="Agents", command=self.toggle_render_all_agents)
        self.render_all_agents_checkbox.place(x=525, y=35)
        self.render_all_episodes_checkbox = tk.Checkbutton(self.frame, width=5, height=1, bg="#AAAAAA",
                                                           text="Episodes", command=self.toggle_render_all_episodes)
        self.render_all_episodes_checkbox.place(x=600, y=35)

        # Status Label1
        self.status1_label_var = tk.StringVar()
        self.status1_label_var.set("-Status1 No output yet-")
        self.status1_label = tk.Label(self.frame, width=93, height=1, bg="#FFFFFF", fg="#000000",
                                      textvariable=self.status1_label_var)
        self.status1_label.place(x=10, y=127)

        # Status Label2
        self.status2_label_var = tk.StringVar()
        self.status2_label_var.set("-Status2 No output yet-")
        self.status2_label = tk.Label(self.frame, width=93, height=1, bg="#FFFFFF", fg="#000000",
                                      textvariable=self.status2_label_var)
        self.status2_label.place(x=10, y=150)
        self.start_time = time()
        self.train_thread = None

    def amount_down_func(self):
        new_amount = self.root.manager_object.max_mutate_amount - 0.05
        self.root.manager_object.max_mutate_amount = new_amount
        self.mutate_amount_label_var.set(str(round(new_amount, ndigits=3)))
        print("\rmax mutate amount =", str(round(new_amount, ndigits=3)), end="")

    def amount_up_func(self):
        new_amount = self.root.manager_object.max_mutate_amount + 0.05
        self.root.manager_object.max_mutate_amount = new_amount
        self.mutate_amount_label_var.set(str(round(new_amount, ndigits=3)))
        print("\rmax mutate amount =", str(round(new_amount, ndigits=3)), end="")

    def chance_down_func(self):
        new_amount = self.root.manager_object.mutate_chance - 0.05
        self.root.manager_object.mutate_chance = new_amount
        self.mutate_chance_label_var.set(str(round(new_amount, ndigits=3)))
        print("\rmutate chance =", str(round(new_amount, ndigits=3)), end="")

    def chance_up_func(self):
        new_amount = self.root.manager_object.mutate_chance + 0.05
        self.root.manager_object.mutate_chance = new_amount
        self.mutate_chance_label_var.set(str(round(new_amount, ndigits=3)))
        print("\rmutate chance =", str(round(new_amount, ndigits=3)), end="")

    def toggle_render_done(self):
        self.root.manager_object.render_done = True if self.root.manager_object.render_done is False else False

    def toggle_render_watchdog(self):
        self.root.manager_object.render_watchdog = True if self.root.manager_object.render_watchdog is False\
            else False

    def toggle_render_best(self):
        self.root.manager_object.render_best = True if self.root.manager_object.render_best is False\
            else False

    def toggle_render_all_agents(self):
        self.root.manager_object.render_all_agents = True if self.root.manager_object.render_all_agents is False\
            else False

    def toggle_render_all_episodes(self):
        self.root.manager_object.render_all_episodes = True if self.root.manager_object.render_all_episodes is False \
            else False

    def save_pop(self, save_name=None):
        save_name = self.root.manager_object.worker_dir+self.root.manager_object.env_name+"_saved_population.p" \
            if save_name is None else save_name
        pickle.dump(self.root.manager_object.agents, open(save_name, "wb"))
        print("population saved")

    def load_pop(self, load_name=None):
        load_name = self.root.manager_object.worker_dir+self.root.manager_object.env_name+"_saved_population.p" \
            if load_name is None else load_name
        self.root.manager_object.agents = pickle.load(open(load_name, "rb"))
        print("population loaded")

    def start_train(self):
        if self.train_thread is not None:
            print("train thread already running")
        else:
            self.train_thread = Thread(name="training_thread", target=self.root.manager_object.train, daemon=True)
            self.train_thread.start()


if __name__ == "__main__":
    manager = AgentManager(population=100, pool_size=0.1, network_layout=[6, 16, 16, 3], env_name="Acrobot-v1",
                           continuous_action=False, episodes_per_agent=5, use_watchdog=-1, watchdog_penalty=0,
                           generations=500, obs_scale=1, action_scale=1)
