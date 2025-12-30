import gym
import numpy as np
import cv2
import math
import os
from gym import spaces

class UUVEnv(gym.Env):
    def __init__(self):
        super(UUVEnv, self).__init__()

        self.width = 1280
        self.height = 720

        self.start_pos = [13, 110]
        self.goal_pos = [1002, 597]
        self.uuv_pos = list(self.start_pos)
        self.prev_pos = list(self.start_pos)

        self.asset_path = os.path.join(os.path.dirname(__file__), 'assets')
        self.water_img = cv2.imread(os.path.join(self.asset_path, 'water.png'))
        self.uuv_img = cv2.imread(os.path.join(self.asset_path, 'uuv.png'), cv2.IMREAD_UNCHANGED)
        if self.water_img is None or self.uuv_img is None:
            raise FileNotFoundError("Assets (water.png or uuv.png) not found in uuv_env/assets directory")

        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self.obstacles = [
            [159, 169, 310, 285], [8, 335, 190, 410], [315, 335, 473, 449], [5, 457, 446, 595],
            [4, 588, 182, 714], [404, 541, 692, 717], [642, 147, 798, 268], [627, 365, 799, 434],
            [885, 250, 1078, 333], [1104, 60, 1273, 142], [1083, 372, 1253, 445], [1096, 576, 1268, 715]
        ]

        self.reset_vars()
        self.current_episode = 0
        self.prev_action = None

    def reset_vars(self):
        self.uuv_pos = list(self.start_pos)
        self.prev_pos = list(self.start_pos)
        self.path_length = 0
        self.time = 0
        self.done = False
        self.rewards = []
        self.distances = []
        self.collisions = []
        self.prev_dist = self._calculate_distance(self.uuv_pos, self.goal_pos)
        self.min_dist_to_goal = self.prev_dist
        self.prev_action = None

    def reset(self):
        self.reset_vars()
        print("Environment reset.")
        return self._get_obs()

    def _get_obs(self):
        dist_to_goal = self._calculate_distance(self.uuv_pos, self.goal_pos)
        angle_to_goal = math.atan2(self.goal_pos[1] - self.uuv_pos[1], self.goal_pos[0] - self.uuv_pos[0])
        x_norm = self.uuv_pos[0] / self.width
        y_norm = self.uuv_pos[1] / self.height
        dist_norm = dist_to_goal / (self.width + self.height)
        angle_norm = (angle_to_goal + np.pi) / (2 * np.pi)

        self.min_dist_to_goal = min(self.min_dist_to_goal, dist_to_goal)

        dist_to_obstacle_up = self._get_distance_to_obstacle(0, -1)
        dist_to_obstacle_down = self._get_distance_to_obstacle(0, 1)
        dist_to_obstacle_left = self._get_distance_to_obstacle(-1, 0)
        dist_to_obstacle_right = self._get_distance_to_obstacle(1, 0)

        max_distance = self.width + self.height
        dist_to_obstacle_up_norm = dist_to_obstacle_up / max_distance
        dist_to_obstacle_down_norm = dist_to_obstacle_down / max_distance
        dist_to_obstacle_left_norm = dist_to_obstacle_left / max_distance
        dist_to_obstacle_right_norm = dist_to_obstacle_right / max_distance

        return np.array([
            x_norm, y_norm, dist_norm, angle_norm,
            dist_to_obstacle_up_norm, dist_to_obstacle_down_norm,
            dist_to_obstacle_left_norm, dist_to_obstacle_right_norm
        ], dtype=np.float32)

    def _calculate_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _get_distance_to_obstacle(self, dx, dy):
        x, y = self.uuv_pos
        distance = 0
        max_distance = self.width + self.height

        while 0 <= x < self.width and 0 <= y < self.height and distance < max_distance:
            x += dx
            y += dy
            distance += 1
            for ox1, oy1, ox2, oy2 in self.obstacles:
                if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                    return distance
        return max_distance

    def _is_collision(self):
        x, y = self.uuv_pos
        for i, (ox1, oy1, ox2, oy2) in enumerate(self.obstacles):
            if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                print(f"Collision detected with obstacle {i}: [{ox1}, {oy1}, {ox2}, {oy2}]")
                return True
        return False

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, {}

        self.prev_pos = list(self.uuv_pos)

        dist_to_goal = self._calculate_distance(self.uuv_pos, self.goal_pos)
        step_size = 1 if dist_to_goal < 50 else 20
        if action == 0: self.uuv_pos[1] -= step_size
        elif action == 1: self.uuv_pos[1] += step_size
        elif action == 2: self.uuv_pos[0] -= step_size
        elif action == 3: self.uuv_pos[0] += step_size
        elif action == 4: pass

        self.uuv_pos[0] = max(0, min(self.width - 40, self.uuv_pos[0]))
        self.uuv_pos[1] = max(0, min(self.height - 40, self.uuv_pos[1]))

        dist_to_goal = self._calculate_distance(self.uuv_pos, self.goal_pos)
        angle_to_goal = math.atan2(self.goal_pos[1] - self.uuv_pos[1], self.goal_pos[0] - self.uuv_pos[0])
        dist_change = self.prev_dist - dist_to_goal
        self.prev_dist = dist_to_goal
        self.path_length += self._calculate_distance(self.prev_pos, self.uuv_pos)
        self.time += 1

        collision = self._is_collision()
        reached = dist_to_goal < 1
        self.done = collision or reached or self.time >= 2000

        if reached:
            self.render()

        obs_dists = self._get_obs()[4:] * (self.width + self.height)
        proximity_penalty = -sum(20 * max(0, 50 - d) for d in obs_dists if d < 50)

        boundary_penalty = 0
        if self.uuv_pos[0] >= self.width - 40 or self.uuv_pos[0] <= 40 or \
           self.uuv_pos[1] >= self.height - 40 or self.uuv_pos[1] <= 40:
            boundary_penalty -= 100

        y_change = self.uuv_pos[1] - self.prev_pos[1]
        y_reward = 0
        if self.uuv_pos[1] < self.goal_pos[1] and y_change > 0:
            y_reward += 150.0
        elif self.uuv_pos[1] > self.goal_pos[1] and y_change < 0:
            y_reward += 150.0

        if action != 4:
            move_angle = math.atan2(self.uuv_pos[1] - self.prev_pos[1], self.uuv_pos[0] - self.prev_pos[0])
            angle_diff = abs(angle_to_goal - move_angle)
            angle_reward = max(0, 20 - angle_diff * 20 / np.pi) if dist_change > 0 else 0
        else:
            angle_reward = -10

        action_penalty = -10 if self.prev_action == action else 0
        self.prev_action = action

        reward = -dist_to_goal * 0.02
        reward += dist_change * 100.0
        reward += proximity_penalty
        reward += y_reward
        reward += angle_reward
        reward += boundary_penalty
        reward += action_penalty
        reward -= self.time * 0.01
        if reached:
            reward += 10000
            print(f"Goal reached at {self.uuv_pos}, Distance: {dist_to_goal:.2f}")
        if collision:
            reward -= 500

        self.rewards.append(reward)
        self.distances.append(dist_to_goal)
        self.collisions.append(collision)

        print(f"Episode: {self.current_episode} | Position: {self.uuv_pos}, Distance: {dist_to_goal:.2f}, Path Length: {self.path_length:.2f}, Time: {self.time}, Reward: {reward:.2f}, Collision: {collision}")

        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        try:
            print("Rendering frame...")
            frame = self.water_img.copy()

            for ox1, oy1, ox2, oy2 in self.obstacles:
                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)

            x, y = self.uuv_pos
            x = max(0, min(self.width - 40, x))
            y = max(0, min(self.height - 40, y))
            uuv_resized = cv2.resize(self.uuv_img, (40, 40))
            alpha_s = uuv_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            y_start = max(0, int(y))
            y_end = min(self.height, int(y + 40))
            x_start = max(0, int(x))
            x_end = min(self.width, int(x + 40))

            uuv_h = y_end - y_start
            uuv_w = x_end - x_start

            if uuv_h > 0 and uuv_w > 0:
                uuv_resized = uuv_resized[:uuv_h, :uuv_w]
                alpha_s = alpha_s[:uuv_h, :uuv_w]
                alpha_l = alpha_l[:uuv_h, :uuv_w]
                for c in range(0, 3):
                    frame[y_start:y_end, x_start:x_end, c] = (alpha_s * uuv_resized[:, :, c] +
                                                              alpha_l * frame[y_start:y_end, x_start:x_end, c])

            cv2.circle(frame, tuple(self.start_pos), 10, (0, 255, 0), -1)
            cv2.circle(frame, tuple(self.goal_pos), 10, (255, 0, 0), -1)

            cv2.imwrite(f'final_frame_episode_{self.current_episode}.png', frame)
            print(f"Saved final_frame_episode_{self.current_episode}.png")
        except Exception as e:
            print(f"Rendering error: {e}")

    def close(self):
        print("Closing environment resources...")