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
import os
import psutil
import socket
# import json
import pickle


class AgentManager:
    def __init__(self, population=50, network_layout=None, env_name=None, continuous_action=True, generations=999,
                 episodes_per_agent=1, pool_size=0.25, obs_scale=1, action_scale=1, input_nodes=2, output_nodes=2,
                 timestep_limit=None, use_watchdog=0, watchdog_penalty=0, n_procs=-1, is_client=False):
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
        self.cross_over_chance = 0.5
        self.mutate_chance = 1.0
        self.random_weight_chance = 0.001
        self.max_mutate_amount = 0.1
        self.population_to_keep = int(self.total_population * pool_size)
        print(f"pool/population size {self.population_to_keep}/{self.total_population}")
        self.model_dir = "models/"
        self.worker_dir = "workers/"
        self.bind_address = "0.0.0.0"
        self.port = 8015
        self.server_address = "192.168.1.10"
        self.server_port = 8015
        self.socket_timeout = 60
        self.buffer_size = 2048
        self.waiting_agent_list = []
        self.current_generation = 0
        self.completed_agents = 0
        self.high_score = -999
        self.is_training = False
        self.client_db = {}
        self.is_client = is_client
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

        print(f"starting {self.n_procs} processes")
        for w in range(self.n_procs):
            proc = Process(name="worker"+str(w), target=self.worker, daemon=True, args=(w,))
            proc.start()
            sleep(0.5)

        self.agents = self.create_population()
        if not self.is_client:
            self.scores = [0.0 for _ in range(self.total_population)]
            self.network_thread = Thread(name="server_thread", target=self.server_thread_function, daemon=True)
            self.network_thread.start()
        else:
            self.network_thread = Thread(name="client_thread", target=self.client_thread_function, daemon=True,
                                         args=[self.server_address, self.server_port])
            self.network_thread.start()

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

    def server_thread_function(self):
        print("starting server thread")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.bind_address, self.port))
        s.listen(10)

        print(f"AI server listening on address {self.bind_address} port {self.port}")

        while True:
            address = ["unknown", 0]
            try:
                client_socket, address = s.accept()
                client_socket.settimeout(self.socket_timeout)
                msg = client_socket.recv(self.buffer_size)
                decoded_msg = msg.decode("utf-8").split(sep=":")
                if decoded_msg[0] == "client":
                    if decoded_msg[1] == "get":
                        self.handle_get(address[0], client_socket)
                    elif decoded_msg[1] == "put":
                        self.handle_put(address[0], client_socket, decoded_msg)
                    else:
                        print(f"client {address[0]} invalid request {decoded_msg}")
                else:
                    print("unknown connection from", address[0], "sending alert response...")
                    client_socket.send(bytes(f"alert {address[0]}", "utf-8"))
                client_socket.close()
            except (ConnectionError, socket.timeout, UnicodeDecodeError) as e:
                print("Error in client communication:", e)
                if address[0] in self.client_db.keys():
                    print(f"removing {address[0]} from client_db - connection issues")
                    for w_id in self.client_db[address[0]]["proc_ids"]:
                        self.waiting_agent_list.append(w_id)
                    del self.client_db[address[0]]

    def handle_get(self, remote_address, client_socket):
        if self.is_training is False or len(self.waiting_agent_list) == 0:
            work_id = -1
            agent_bytes = bytes()
        else:
            work_id = self.waiting_agent_list.pop(0)
            agent_bytes = pickle.dumps(self.agents[work_id])
        agent_header = f"ok:{work_id}:{len(agent_bytes)}:{self.current_generation}"
        client_socket.send(bytes(agent_header, "utf-8"))
        header_resp = client_socket.recv(self.buffer_size).decode("utf-8")
        if header_resp == str(len(agent_bytes)):
            if len(agent_bytes) > 0:
                client_socket.sendall(agent_bytes)
                transfer_resp = client_socket.recv(self.buffer_size).decode("utf-8")
                if transfer_resp == str(len(agent_bytes)):
                    client_socket.send(bytes(f"ok:{self.high_score}", "utf-8"))
                    if remote_address in self.client_db.keys():
                        self.client_db[remote_address]["last_seen"] = time()
                        self.client_db[remote_address]["proc_ids"].append(work_id)
                    else:
                        self.client_db[remote_address] = {"last_seen": time(), "proc_ids": [work_id]}
            else:
                client_socket.send(bytes(f"ok:{self.high_score}", "utf-8"))
        elif header_resp == "closing":
            print("client ended connection early")
        else:
            print(f"client {remote_address} unknown response to agent header - '{header_resp}'")

    def client_get_work(self, server_address, server_port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_address, server_port))
            s.settimeout(self.socket_timeout)
            s.send(bytes("client:get", "utf-8"))
            msg = s.recv(self.buffer_size)
            header_msg = msg.decode("utf-8").split(sep=":")
            work_id = int(header_msg[1])
            agent_bytes_length = int(header_msg[2])
            generation = int(header_msg[3])
            if header_msg[0] == "ok":
                s.send(bytes(str(agent_bytes_length), "utf-8"))
                if agent_bytes_length > 0:
                    print(f"getting {agent_bytes_length} bytes for agent {work_id}")
                    agent_bytes = bytes()
                    data_counter = 0
                    while data_counter < agent_bytes_length:
                        agent_bytes += s.recv(102400)
                        data_counter = len(agent_bytes)
                    s.send(bytes(str(len(agent_bytes)), "utf-8"))
                    resp = s.recv(self.buffer_size).decode("utf-8")
                    resp = resp.split(sep=":")
                    if resp[0] == "ok":
                        data = pickle.loads(agent_bytes)
                        self.agents[work_id] = data
                        # for a in range(len(data)):
                        #     agent = Agent(agent_id=a, network_layout=self.network_layout,
                        #                   continuous_action=self.continuous_action,
                        #                   input_nodes=self.input_nodes, output_nodes=self.output_nodes)
                        #     agent.network = data[a][0]
                        #     agent.biases = data[a][1]
                        #     self.agents.append(agent)

                        self.high_score = float(resp[1])
                        self.current_generation = generation
                        print(f"-> gen:{generation} agent:{work_id} score:{round(self.high_score, ndigits=2)}")
                        self.update_status_line_2(f"gen:{generation} high score:{round(self.high_score, ndigits=2)}")
                        return work_id
        except (ConnectionError, socket.timeout, TimeoutError, OSError) as e:
            print("server seems unavailable at", server_address, e)
        return -1

    def handle_put(self, remote_address, client_socket, client_msg):
        # print(f"client {remote_address} agent {client_msg[2]} score {client_msg[3]} generation {client_msg[4]}")
        a_id = int(client_msg[2])
        score = float(client_msg[3])
        generation = int(client_msg[4])

        if remote_address not in self.client_db.keys():
            print(f"client {remote_address} not in client_db")
            client_socket.send(bytes("invalid_client", "utf-8"))
        elif generation != self.current_generation:
            print(f"client {remote_address} sent expired results for gen: {generation} agent: {a_id}")
            client_socket.send(bytes("expired", "utf-8"))
            if a_id in self.client_db[remote_address]["proc_ids"]:
                self.client_db[remote_address]["proc_ids"].remove(a_id)
                self.waiting_agent_list.append(a_id)
        elif remote_address in self.client_db.keys() and a_id not in self.client_db[remote_address]["proc_ids"]:
            print(f"client {remote_address} agent {a_id} is invalid {self.client_db[remote_address]['proc_ids']}")
            client_socket.send(bytes("invalid_work", "utf-8"))
        elif remote_address in self.client_db.keys() and a_id in self.client_db[remote_address]["proc_ids"]:
            self.agents[a_id].score = score
            self.scores[a_id] = score
            self.completed_agents += 1
            client_socket.send(bytes("ok", "utf-8"))
            self.client_db[remote_address]["proc_ids"].remove(a_id)
        else:
            print(f"client {remote_address} sent invalid score data'{client_msg}'")
            client_socket.send(bytes("invalid_score", "utf-8"))

    def client_send_result(self, server_address, server_port, agent_id, score, generation):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((server_address, server_port))
            s.settimeout(self.socket_timeout)
            s.send(bytes(f"client:put:{agent_id}:{score}:{generation}", "utf-8"))
            msg = s.recv(self.buffer_size)
            decoded_msg = msg.decode("utf-8")
            if decoded_msg == "ok":
                return 0
            else:
                print(f"sending score failed. server responded with {decoded_msg}")
                return 1
        except (ConnectionError, socket.timeout, TimeoutError, OSError) as e:
            print("server seems unavailable at", server_address, e)
            return 1

    def client_thread_function(self, server_address, server_port):
        print("starting client thread")
        proc_ids = [-1 for _ in range(self.n_procs)]
        last_ui_update_time = time()
        while True:
            is_training = True
            while is_training is True:
                for proc_id in range(len(proc_ids)):
                    if proc_ids[proc_id] == -1:
                        agent_index = self.client_get_work(server_address, server_port)
                        if agent_index != -1:
                            proc_ids[proc_id] = agent_index
                            do_render = True if self.render_all_agents else False
                            pickle.dump([self.agents[agent_index],
                                         {"agent_id": agent_index, "eps": self.episodes_per_agent,
                                          "do_render": do_render, "render_watchdog": self.render_watchdog,
                                          "render_done": self.render_done, "all_eps": self.render_all_episodes}],
                                        open(f"{self.worker_dir}worker{proc_id}", "wb"))
                        else:
                            if proc_ids == [-1, -1, -1, -1]:
                                is_training = False
                                break
                    else:
                        try:
                            result = pickle.load(open(f"{self.worker_dir}worker{proc_id}result", "rb"))
                            if result is not None:
                                self.client_send_result(server_address, server_port, proc_ids[proc_id], result,
                                                        self.current_generation)
                                proc_ids[proc_id] = -1
                                self.remove_file(f"{self.worker_dir}worker{proc_id}result")
                            else:
                                do_render = True if self.render_all_agents else False
                                pickle.dump([self.agents[proc_ids[proc_id]],
                                             {"agent_id": proc_ids[proc_id], "eps": self.episodes_per_agent,
                                              "do_render": do_render, "render_watchdog": self.render_watchdog,
                                              "render_done": self.render_done, "all_eps": self.render_all_episodes}],
                                            open(f"{self.worker_dir}worker{proc_id}", "wb"))
                        except (FileNotFoundError, EOFError):
                            pass
                sleep(0.1)
                if time() - last_ui_update_time > 1:
                    last_ui_update_time = time()
                    self.update_status_line_1(self.current_generation, proc_ids)
            sleep(2.0)

    def train(self):
        print("begin training...")
        self.is_training = True
        self.scores = [0.0 for _ in range(self.total_population)]
        best_agents = []
        last_ui_update_time = time()
        for e in range(self.generations):
            self.current_generation = e
            self.waiting_agent_list = [a for a in range(len(self.agents))]
            self.completed_agents = 0
            proc_ids = [-1 for _ in range(self.n_procs)]
            while self.completed_agents < self.total_population:
                for proc_id in range(self.n_procs):
                    if len(self.waiting_agent_list) != 0 and proc_ids[proc_id] == -1:
                        next_agent = self.waiting_agent_list.pop(0)
                        do_render = True if next_agent in best_agents and self.render_best \
                            or self.render_all_agents \
                            else False
                        proc_ids[proc_id] = next_agent
                        pickle.dump([self.agents[next_agent],
                                     {"agent_id": next_agent, "eps": self.episodes_per_agent,
                                      "do_render": do_render, "render_watchdog": self.render_watchdog,
                                      "render_done": self.render_done, "all_eps": self.render_all_episodes}],
                                    open(self.worker_dir+"worker"+str(proc_id), "wb"))
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
                                # print(f"worker{proc_id} gave score {agent_score} for agent {proc_ids[proc_id]} "
                                #       f"generation {self.current_generation}")

                                self.scores[proc_ids[proc_id]] = agent_score
                                self.agents[proc_ids[proc_id]].score = agent_score
                                self.completed_agents += 1
                                proc_ids[proc_id] = -1
                                self.remove_file(self.worker_dir+"worker"+str(proc_id)+"result")
                        except (FileNotFoundError, EOFError):
                            pass
                sleep(0.1)
                for key in list(self.client_db.keys()):
                    if time() - self.client_db[key]["last_seen"] > 90:
                        for client_work_id in self.client_db[key]["proc_ids"]:
                            self.waiting_agent_list.append(client_work_id)
                        print(f"removing {key} from client_db - inactive")
                        del self.client_db[key]
                if time() - last_ui_update_time > 1:
                    self.update_status_line_1(e, proc_ids)
                    last_ui_update_time = time()
            best_agents = self.process_agents(e)
            self.scores = [0.0 for _ in range(self.total_population)]
            for a in self.agents:
                a.reset()
        self.frontend.control_window.train_thread = None
        self.is_training = False

    def update_status_line_1(self, gen, current_agents):
        running_local_agents = [x for x in current_agents]
        running_remote_agents = []
        for key in self.client_db.keys():
            for p_id in self.client_db[key]["proc_ids"]:
                running_remote_agents.append(p_id)
        gen_for_str = gen if gen is not None else -1
        a_status = f"gen: {gen_for_str+1} completed {self.completed_agents} " \
                   f"running agents: {running_local_agents}{running_remote_agents}"
        self.frontend.control_window.status1_label_var.set(a_status)

    def update_status_line_2(self, status_string):
        self.frontend.control_window.status2_label_var.set(status_string)

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
            if self.use_watchdog > 1 and timestep != 0 and timestep % self.use_watchdog == 0:
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
        self.high_score = self.scores[0]
        for i in range(self.population_to_keep):
            shuffled_list = list(range(len(self.agents)))
            shuffle(shuffled_list)
            for agent_i in shuffled_list:
                _agent = self.agents[agent_i]
                if _agent.score == self.scores[i] and _agent.agent_id not in best_agents:
                    best_agents.append(_agent.agent_id)
                    break
        e_stat = f"gen:{epoch+1} avg:{round(overall_avg_score, ndigits=2)} high:{round(self.high_score, ndigits=2)}" \
                 f" top 5: {best_agents[:5]} best layout: {self.agents[best_agents[0]].network_layout}"
        print(e_stat)
        self.update_status_line_2(e_stat)

        # all_agents = list(range(self.total_population))
        for p in range(self.total_population):
            if p not in best_agents:
                if random() < self.cross_over_chance:
                    self.agents[p].network, self.agents[p].biases = self.cross_over(best_agents, best_agents)
                    self.agents[p].network, self.agents[p].biases = self.mutate_network([p])
                else:
                    self.agents[p].network, self.agents[p].biases = self.mutate_network(best_agents)
        return best_agents

    def cross_over(self, agent_list_a, agent_list_b):
        net_a = deepcopy(self.agents[choice(agent_list_a)].network)
        net_a_bias = deepcopy(self.agents[choice(agent_list_a)].biases)
        net_b = self.agents[choice(agent_list_b)].network
        net_b_bias = self.agents[choice(agent_list_b)].biases

        for layer_index, layer in enumerate(net_a):
            for node_index, node in enumerate(layer):
                for weight_index, weight in enumerate(node):
                    if random() >= 0.5:
                        net_a[layer_index][node_index][weight_index] = net_b[layer_index][node_index][weight_index]
            for bias_index, bias in enumerate(net_a_bias[layer_index]):
                if random() >= 0.5:
                    net_a_bias[layer_index][bias_index] = net_b_bias[layer_index][bias_index]
        return deepcopy(net_a), deepcopy(net_a_bias)

    @staticmethod
    def random_float():
        return (random() - 0.5) * 2

    def mutate_network(self, agent_list):
        chosen_one = choice(agent_list)
        chosen_network = deepcopy(self.agents[chosen_one].network)
        chosen_biases = deepcopy(self.agents[chosen_one].biases)
        for layer in range(len(chosen_network)):
            for node in range(len(chosen_network[layer])):
                for weight in range(len(chosen_network[layer][node])):
                    if random() < self.mutate_chance:
                        if random() < self.random_weight_chance:
                            _wt = self.random_float()
                        else:
                            _wt = chosen_network[layer][node][weight] + (self.random_float() * self.max_mutate_amount)
                        chosen_network[layer][node][weight] = float(np.clip(_wt, -1.0, 1.0))
            for bias_index in range(len(chosen_biases[layer])):
                if 0 <= layer < len(chosen_network) - 1 and random() < self.mutate_chance:
                    if random() < self.random_weight_chance:
                        _bias = self.random_float()
                    else:
                        _bias = chosen_biases[layer][bias_index] + (self.random_float() * self.max_mutate_amount)
                    # noinspection PyTypeChecker
                    chosen_biases[layer][bias_index] = float(np.clip(_bias, -1.0, 1.0))
        return deepcopy(chosen_network), deepcopy(chosen_biases)


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
        self.max_layers = 3
        self.min_layer_nodes = 8
        self.max_layer_nodes = 16
        self.network_layout = network_layout
        self.network, self.biases = self.initialize_network()

    def initialize_network(self):
        if self.network_layout is None:
            new_layout = [self.input_nodes]
            new_layers = int(randrange(self.min_layers, self.max_layers + 1))
            for _new_layer in range(new_layers):
                new_layout.append(int(randrange(self.min_layer_nodes, self.max_layer_nodes + 1)))
            new_layout.append(self.output_nodes)
            self.network_layout = new_layout

        finished_network = []
        finished_biases = []
        for layer in range(len(self.network_layout) - 1):
            layer_nodes = []
            for _node in range(self.network_layout[layer]):
                weights_per_node = []
                for weight in range(self.network_layout[layer + 1]):
                    weights_per_node.append(random() - 0.5)
                layer_nodes.append(weights_per_node)
            finished_network.append(layer_nodes)
            if layer == (len(self.network_layout) - 1) - 1:
                layer_biases = [0 for _ in range(len(layer_nodes[-1]))]
            else:
                layer_biases = [(random() - 0.5) for _ in range(len(layer_nodes[-1]))]
            finished_biases.append(layer_biases)
        return deepcopy(finished_network), deepcopy(finished_biases)

    def predict(self, obs):
        return self.calculate_action(obs)

    def calculate_action(self, input_layer):
        weights = self.network
        biases = self.biases
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
                bias = biases[layer][index]
                input_layer.append(tanh(activation+bias))
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

        if not self.root.manager_object.is_client:
            # The Start button (don't touch it)
            self.start_button = tk.Button(self.frame, width=5, bg="#CCCCCC", text="START", command=self.start_train)
            self.start_button.place(x=10, y=10)

            # Mutate controls
            self.mutate_chance_title = tk.Label(self.frame, width=11, bg="#AAAAAA", text="Mutate Chance")
            self.mutate_chance_title.place(x=10, y=52)
            self.chance_down = tk.Button(self.frame, width=1, bg="#CCCCCC", text="-", command=self.chance_down_func)
            self.chance_down.place(x=110, y=50)
            self.mutate_chance_label_var = tk.StringVar()
            self.mutate_chance_label_var.set(str(self.root.manager_object.mutate_chance))
            self.mutate_chance_label = tk.Label(self.frame, width=5, textvariable=self.mutate_chance_label_var)
            self.mutate_chance_label.place(x=128, y=52)
            self.chance_up = tk.Button(self.frame, width=1, bg="#CCCCCC", text="+", command=self.chance_up_func)
            self.chance_up.place(x=170, y=50)

            self.mutate_amount_title = tk.Label(self.frame, width=11, bg="#AAAAAA", text="Mutate Amount")
            self.mutate_amount_title.place(x=10, y=92)
            self.mutate_down = tk.Button(self.frame, width=1, bg="#CCCCCC", text="-", command=self.amount_down_func)
            self.mutate_down.place(x=110, y=90)
            self.mutate_amount_label_var = tk.StringVar()
            self.mutate_amount_label_var.set(str(self.root.manager_object.max_mutate_amount))
            self.mutate_amount_label = tk.Label(self.frame, width=5, textvariable=self.mutate_amount_label_var)
            self.mutate_amount_label.place(x=128, y=92)
            self.mutate_up = tk.Button(self.frame, width=1, bg="#CCCCCC", text="+", command=self.amount_up_func)
            self.mutate_up.place(x=170, y=90)

            # Save and Load buttons
            self.save_button = tk.Button(self.frame, width=5, text="SAVE", bg="#CCCCCC", command=self.save_pop)
            self.save_button.place(x=150, y=10)
            self.load_button = tk.Button(self.frame, width=5, text="LOAD", bg="#CCCCCC", command=self.load_pop)
            self.load_button.place(x=210, y=10)

        # Render options
        self.render_title = tk.Label(self.frame, width=11, bg="#AAAAAA", text="Render Options")
        self.render_title.place(x=440, y=12)
        self.render_done_checkbox = tk.Checkbutton(self.frame, width=5, height=1, bg="#AAAAAA", text="Done",
                                                   command=self.toggle_render_done)
        self.render_done_checkbox.place(x=310, y=35)
        self.render_watchdog_checkbox = tk.Checkbutton(self.frame, width=7, height=1, bg="#AAAAAA", text="Watchdog",
                                                       command=self.toggle_render_watchdog)
        self.render_watchdog_checkbox.place(x=375, y=35)
        self.render_best_checkbox = tk.Checkbutton(self.frame, width=3, height=1, bg="#AAAAAA", text="Best 5",
                                                   command=self.toggle_render_best)
        self.render_best_checkbox.place(x=465, y=35)
        self.render_all_agents_checkbox = tk.Checkbutton(self.frame, width=5, height=1, bg="#AAAAAA", text="Agents",
                                                         command=self.toggle_render_all_agents)
        self.render_all_agents_checkbox.place(x=525, y=35)
        self.render_all_episodes_checkbox = tk.Checkbutton(self.frame, width=5, height=1, bg="#AAAAAA", text="Episodes",
                                                           command=self.toggle_render_all_episodes)
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
        mod_amount = 0.005 if self.root.manager_object.max_mutate_amount < 0.11 else 0.05
        new_amount = self.root.manager_object.max_mutate_amount - mod_amount
        self.root.manager_object.max_mutate_amount = new_amount
        self.mutate_amount_label_var.set(str(round(new_amount, ndigits=3)))
        print("\rmax mutate amount =", str(round(new_amount, ndigits=3)), end="")

    def amount_up_func(self):
        mod_amount = 0.005 if self.root.manager_object.max_mutate_amount < 0.1 else 0.05
        new_amount = self.root.manager_object.max_mutate_amount + mod_amount
        self.root.manager_object.max_mutate_amount = new_amount
        self.mutate_amount_label_var.set(str(round(new_amount, ndigits=3)))
        print("\rmax mutate amount =", str(round(new_amount, ndigits=3)), end="")

    def chance_down_func(self):
        mod_amount = 0.005 if self.root.manager_object.mutate_chance < 0.11 else 0.05
        new_amount = self.root.manager_object.mutate_chance - mod_amount
        self.root.manager_object.mutate_chance = new_amount
        self.mutate_chance_label_var.set(str(round(new_amount, ndigits=3)))
        print("\rmutate chance =", str(round(new_amount, ndigits=3)), end="")

    def chance_up_func(self):
        mod_amount = 0.005 if self.root.manager_object.mutate_chance < 0.1 else 0.05
        new_amount = self.root.manager_object.mutate_chance + mod_amount
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
        save_name = self.root.manager_object.model_dir+self.root.manager_object.env_name+"_saved_population.p" \
            if save_name is None else save_name
        pickle.dump(self.root.manager_object.agents, open(save_name, "wb"))
        print("population saved")

    def load_pop(self, load_name=None):
        load_name = self.root.manager_object.model_dir+self.root.manager_object.env_name+"_saved_population.p" \
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
    manager = AgentManager(population=100, pool_size=0.1, network_layout=[24, 16, 16, 4], env_name="BipedalWalkerHardcore-v3",
                           continuous_action=True, episodes_per_agent=10, generations=500, use_watchdog=300, watchdog_penalty=100,
                           obs_scale=1, action_scale=1,
                           n_procs=4)
