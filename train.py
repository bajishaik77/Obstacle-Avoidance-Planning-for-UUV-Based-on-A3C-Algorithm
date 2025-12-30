import os
import tensorflow as tf
import threading
import matplotlib.pyplot as plt
import numpy as np
import cv2
from a3c.model import ACNet
from a3c.worker import Worker

tf.compat.v1.disable_v2_behavior()

MAX_GLOBAL_EP = 5000
GLOBAL_EP = [0]
GLOBAL_REWARDS = []
SUCCESS_RATES = []
AVG_MIN_DISTANCES = []
ALL_POSITIONS = []
ALL_ACTIONS = []
WINDOW_SUCCESS_RATES = []

def main():
    global GLOBAL_EP, GLOBAL_REWARDS, SUCCESS_RATES, AVG_MIN_DISTANCES, ALL_POSITIONS, ALL_ACTIONS, WINDOW_SUCCESS_RATES
    global_net_scope = 'Global_Net'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    stop_event = threading.Event()
    workers = []

    try:
        global_net = ACNet(global_net_scope, sess)
        for worker_id in range(3):
            worker = Worker(
                worker_id, global_net, sess, gamma=0.99,
                entropy_beta=0.5, max_grad_norm=40,
                global_ep=GLOBAL_EP,
                global_rewards=GLOBAL_REWARDS,
                success_rates=SUCCESS_RATES,
                avg_min_distances=AVG_MIN_DISTANCES,
                all_positions=ALL_POSITIONS,
                all_actions=ALL_ACTIONS,
                max_global_ep=MAX_GLOBAL_EP,
                stop_event=stop_event,
                window_success_rates=WINDOW_SUCCESS_RATES
            )
            workers.append(worker)

        sess.run(tf.compat.v1.global_variables_initializer())

        if not os.path.exists('models'):
            os.makedirs('models')

        saver = tf.compat.v1.train.Saver(max_to_keep=5)

        print("Training from scratch, ignoring any existing checkpoints.")

        worker_threads = []
        for worker in workers:
            t = threading.Thread(target=worker.work)
            t.start()
            worker_threads.append(t)

        # Monitor threads without Coordinator
        while GLOBAL_EP[0] < MAX_GLOBAL_EP and not stop_event.is_set():
            alive = [t.is_alive() for t in worker_threads]
            if not any(alive):
                break
            if GLOBAL_EP[0] % 100 == 0 and GLOBAL_EP[0] > 0:
                saver.save(sess, 'models/uuv_model.ckpt', global_step=GLOBAL_EP[0])
                print(f"Checkpoint saved at episode {GLOBAL_EP[0]}")
            for t in worker_threads:
                t.join(timeout=1.0)  # Non-blocking join with timeout

        saver.save(sess, 'models/uuv_model_final.ckpt')

        if GLOBAL_REWARDS:
            window_size = 100
            moving_avg_rewards = []
            for i in range(len(GLOBAL_REWARDS)):
                start = max(0, i - window_size + 1)
                window = GLOBAL_REWARDS[start:i+1]
                moving_avg_rewards.append(np.mean(window))

            plt.figure(figsize=(12, 8))
            plt.plot(GLOBAL_REWARDS, label='Raw Rewards', alpha=0.3)
            plt.plot(moving_avg_rewards, label='Moving Average (window=100)', color='red')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Moving Average of Episode Rewards')
            plt.legend()
            plt.grid(True)
            plt.savefig('moving_avg_reward_plot.png')
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(range(1, len(WINDOW_SUCCESS_RATES) + 1), WINDOW_SUCCESS_RATES, label='Window Success Rate (%)')
            plt.xlabel('Episode')
            plt.ylabel('Window Success Rate (%)')
            plt.ylim(0, 100)
            plt.title('Window Success Rate Over Time (Window Size = 40)')
            plt.legend()
            plt.grid(True)
            plt.savefig('window_success_rate_plot.png')
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(range(1, len(AVG_MIN_DISTANCES) + 1), AVG_MIN_DISTANCES, label='Avg Min Distance to Goal')
            plt.xlabel('Episode')
            plt.ylabel('Distance')
            plt.title('Average Minimum Distance to Goal vs. Episode')
            plt.legend()
            plt.grid(True)
            plt.savefig('distance_to_goal_plot.png')
            plt.close()

            plt.figure(figsize=(12, 8))
            target_episode_idx = None
            for i, dist in enumerate(AVG_MIN_DISTANCES):
                if dist < 1:
                    target_episode_idx = i
                    break
            for i, positions in enumerate(ALL_POSITIONS[-5:], start=max(0, len(ALL_POSITIONS)-5)):
                if positions:
                    x, y = zip(*positions)
                    if target_episode_idx is not None and i == target_episode_idx:
                        plt.plot(x, y, label=f'Episode {i+1} (Target Reached)', linestyle=':', linewidth=2)
                    else:
                        plt.plot(x, y, label=f'Episode {i+1}')
            plt.scatter([1002], [597], color='red', marker='*', s=200, label='Goal [1002, 597]')
            plt.scatter([13], [110], color='green', marker='o', s=200, label='Start [13, 110]')
            for ox1, oy1, ox2, oy2 in workers[0].env.obstacles:
                plt.gca().add_patch(plt.Rectangle((ox1, oy1), ox2-ox1, oy2-oy1, fill=True, color='gray', alpha=0.5))
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('A3C Collision Avoidance Trajectory')
            plt.legend()
            plt.grid(True)
            plt.savefig('path_trajectories_with_obstacles.png')
            plt.close()

            water_img = cv2.imread(os.path.join(workers[0].env.asset_path, 'water.png'))
            if water_img is None:
                print("Failed to load water.png for path plot")
            else:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(water_img, cv2.COLOR_BGR2RGB))
                target_episode_idx = None
                for i, dist in enumerate(AVG_MIN_DISTANCES):
                    if dist < 1:
                        target_episode_idx = i
                        break
                for i, positions in enumerate(ALL_POSITIONS[-5:], start=max(0, len(ALL_POSITIONS)-5)):
                    if positions:
                        x, y = zip(*positions)
                        if target_episode_idx is not None and i == target_episode_idx:
                            plt.plot(x, y, label=f'Episode {i+1} (Target Reached)', linestyle=':', linewidth=2)
                        else:
                            plt.plot(x, y, label=f'Episode {i+1}')
                plt.scatter([1002], [597], color='red', marker='*', s=200, label='Goal [1002, 597]')
                plt.scatter([13], [110], color='green', marker='o', s=200, label='Start [13, 110]')
                for ox1, oy1, ox2, oy2 in workers[0].env.obstacles:
                    plt.gca().add_patch(plt.Rectangle((ox1, oy1), ox2-ox1, oy2-oy1, fill=True, color='gray', alpha=0.5))
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.title('Path Followed by UUV on Water Background')
                plt.legend()
                plt.grid(True)
                plt.savefig('water_path.png')
                plt.close()

            action_counts = np.bincount(ALL_ACTIONS, minlength=5)
            action_labels = ['Up', 'Down', 'Left', 'Right', 'Stay']
            plt.figure(figsize=(12, 8))
            plt.bar(action_labels, action_counts)
            plt.xlabel('Action')
            plt.ylabel('Frequency')
            plt.title('Action Distribution')
            plt.grid(True)
            plt.savefig('action_distribution_plot.png')
            plt.close()

            success_rate = (sum(1 for d in AVG_MIN_DISTANCES if d < 1) / len(AVG_MIN_DISTANCES)) * 100 if AVG_MIN_DISTANCES else 0
            avg_min_distance = np.mean(AVG_MIN_DISTANCES) if AVG_MIN_DISTANCES else float('inf')
            avg_reward = np.mean(GLOBAL_REWARDS) if GLOBAL_REWARDS else 0
            avg_path_length = np.mean([workers[0].env.path_length for _ in range(len(GLOBAL_REWARDS))]) if GLOBAL_REWARDS else 0
            avg_execution_time = np.mean([workers[0].env.time for _ in range(len(GLOBAL_REWARDS))]) if GLOBAL_REWARDS else 0

            print(f"Final Training Metrics: Success Rate: {success_rate:.2f}%, Avg Min Distance: {avg_min_distance:.2f}, Avg Reward: {avg_reward:.2f}")
            print("\nTable: Performance in water.png Environment")
            print("| Algorithm | Path Length (m) | Execution Time (ms) | Target Reached |")
            print("|-----------|-----------------|---------------------|----------------|")
            print(f"| A3C       | {avg_path_length:.1f}           | {avg_execution_time:.1f}            | {'✓' if success_rate >= 80 else '×'}      |")

    except KeyboardInterrupt:
        print("Training interrupted by user. Stopping workers...")
        stop_event.set()
        for t in worker_threads:
            t.join()
        print("All workers stopped.")

    finally:
        for worker in workers:
            worker.env.close()
        sess.close()
        print("All resources cleaned up.")

if __name__ == "__main__":
    main()