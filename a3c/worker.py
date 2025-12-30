import numpy as np
import tensorflow as tf
from a3c.model import ACNet
from uuv_env.uuv_env import UUVEnv

tf.compat.v1.disable_v2_behavior()

class Worker:
    def __init__(self, name, globalAC, sess, gamma=0.99, entropy_beta=0.5, max_grad_norm=40, global_ep=0, global_rewards=[], success_rates=[], avg_min_distances=[], all_positions=[], all_actions=[], max_global_ep=5000, stop_event=None, window_success_rates=[]):
        self.name = f'Worker_{name}'
        self.env = UUVEnv()
        self.sess = sess
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.max_grad_norm = max_grad_norm
        self.AC = ACNet(self.name, sess, globalAC)
        self.episode_rewards = []
        self.global_ep = global_ep
        self.global_rewards = global_rewards
        self.success_rates = success_rates
        self.avg_min_distances = avg_min_distances
        self.all_positions = all_positions
        self.all_actions = all_actions
        self.max_global_ep = max_global_ep
        self.stop_event = stop_event
        self.window_success_rates = window_success_rates
        self.step_count = 0
        self.success_count = 0
        self.min_distances = []
        self.success_window = []
        self.reward_window = []
        self.WINDOW_SIZE = 40
        self.current_episode_positions = []
        self.current_episode = 0
        self.prev_action = None

    def work(self):
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        s = self.env.reset()

        while self.global_ep[0] < self.max_global_ep and not self.stop_event.is_set():
            self.current_episode = self.global_ep[0] + 1
            self.env.current_episode = self.current_episode
            print(f"\n--- Starting Episode {self.current_episode} ---\n")
            self.current_episode_positions = []

            while True:
                try:
                    a = self.AC.choose_action(s)
                except tf.errors.CancelledError:
                    print(f"{self.name}: Session closed, stopping worker.")
                    break

                s_, r, done, info = self.env.step(a)

                self.current_episode_positions.append(self.env.uuv_pos.copy())
                self.all_actions.append(a)

                dist_to_goal = self.env._calculate_distance(self.env.uuv_pos, self.env.goal_pos)
                if dist_to_goal < 1:
                    print(f"Target reached at episode {self.current_episode}, Position: {self.env.uuv_pos}, Distance: {dist_to_goal:.2f}")

                self.step_count += 1

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                ep_r += r

                if total_step % 5 == 0 or done:
                    try:
                        if done:
                            v_s_ = 0
                        else:
                            v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]

                        buffer_v_target = []
                        for rwd in buffer_r[::-1]:
                            v_s_ = rwd + self.gamma * v_s_
                            buffer_v_target.append(v_s_)
                        buffer_v_target.reverse()

                        bs = np.vstack(buffer_s)
                        ba = np.array(buffer_a)
                        bv = np.vstack(buffer_v_target)

                        feed_dict = {
                            self.AC.s: bs,
                            self.AC.a_his: ba,
                            self.AC.v_target: bv,
                        }

                        self.AC.update_global(feed_dict)
                        self.AC.pull_global()
                        buffer_s, buffer_a, buffer_r = [], [], []
                    except tf.errors.CancelledError:
                        print(f"{self.name}: Session closed during update, stopping worker.")
                        break

                s = s_
                total_step += 1

                if done:
                    self.episode_rewards.append(ep_r)
                    self.global_rewards.append(ep_r)
                    self.global_ep[0] += 1

                    success = self.env._calculate_distance(self.env.uuv_pos, self.env.goal_pos) < 1
                    if success:
                        self.success_count += 1
                    self.min_distances.append(self.env.min_dist_to_goal)

                    self.success_window.append(1 if success else 0)
                    self.reward_window.append(ep_r)
                    if len(self.success_window) > self.WINDOW_SIZE:
                        self.success_window.pop(0)
                        self.reward_window.pop(0)

                    success_rate = (self.success_count / self.global_ep[0]) * 100
                    avg_min_distance = np.mean(self.min_distances) if self.min_distances else float('inf')
                    window_success_rate = (sum(self.success_window) / len(self.success_window)) * 100 if self.success_window else 0
                    avg_reward = np.mean(self.reward_window) if self.reward_window else 0

                    self.success_rates.append(success_rate)
                    self.avg_min_distances.append(avg_min_distance)
                    self.window_success_rates.append(window_success_rate)
                    self.all_positions.append(self.current_episode_positions)

                    print(f"{self.name} | Episode: {self.current_episode} | Training: {(self.global_ep[0] / self.max_global_ep) * 100:.2f}% | "
                          f"Success Rate: {success_rate:.2f}% | Window Success Rate: {window_success_rate:.2f}% | "
                          f"Avg Min Distance: {avg_min_distance:.2f} | Avg Reward: {avg_reward:.2f}")

                    if self.global_ep[0] >= self.WINDOW_SIZE and window_success_rate >= 80:
                        print(f"Stopping: Window success rate {window_success_rate:.2f}% met")
                        self.stop_event.set()
                        break

                    s = self.env.reset()
                    ep_r = 0
                    total_step = 1
                    self.prev_action = None
                    break