import torch.multiprocessing as mp
import argparse
from pathlib import Path

from rl_algorithms.flatland_rl.models.a3c import A3CMaster
from rl_algorithms.flatland_rl.models.sac import SACMaster
from rl_algorithms.flatland_rl.utils import set_random_seeds, plot_training_results, get_device


ENV_CONFIG = {
    'width': 50,
    'height': 50,
    'n_agents': 1,
    'max_num_cities': 2
}


def train_sac(args, env_config=None):
    """
    Train a SAC agent in the Flatland environment.

    Parameters:
        args: Parsed command line arguments containing training parameters
        env_config: Parsed environment configuration
    """
    if env_config is None:
        env_config = ENV_CONFIG

    # Initialize SAC agent
    master = SACMaster(env_config)

    # Training loop
    print("Starting SAC training...")
    print(f"Using device: {get_device()}")

    try:
        # Train the agent
        master.train(num_episodes=args.num_episodes)

        # Save final model
        if args.save_model:
            save_path = Path(args.save_dir) / 'sac_final.pth'
            master.save_model(save_path)
            print(f"Model saved to {save_path}")

        # Plot results
        plot_training_results(master.rewards_history, master.steps_history)

        # Evaluate final performance
        eval_rewards = master.evaluate(num_episodes=100)
        print(f"\nFinal evaluation over 100 episodes:")
        print(f"Average reward: {eval_rewards:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    return master


def train_a3c(args, env_config=None):
    """
    Train an A3C agent in the Flatland environment.

    Parameters:
        args: Parsed command line arguments containing training parameters
        env_config: Parsed environment configuration
    """
    if env_config is None:
        env_config = ENV_CONFIG
    master = A3CMaster(env_config)

    # Training loop
    print("Starting A3C training...")
    print(f"Using device: {get_device()}")

    try:
        # Train with multiple workers
        master.train(
            num_episodes=args.num_episodes,
            num_workers=args.num_workers
        )

        # Save final model
        if args.save_model:
            save_path = Path(args.save_dir) / 'a3c_final.pth'
            master.save_model(save_path)
            print(f"Model saved to {save_path}")

        # Plot results
        plot_training_results(master.rewards_history, master.steps_history)

        # Evaluate final performance
        eval_rewards = master.evaluate(num_episodes=100)
        print(f"\nFinal evaluation over 100 episodes:")
        print(f"Average reward: {eval_rewards:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    return master


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='A3C Training in Flatland')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of training episodes')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of A3C workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pretrained model')

    args = parser.parse_args()

    # Create save directory if needed
    if args.save_model:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    set_random_seeds(args.seed)

    # Enable multiprocessing for PyTorch
    if __name__ == '__main__':
        mp.set_start_method('spawn', force=True)
        print(f"Starting training with {args.num_workers} workers...")
        master = train_sac(args)
        print("\nTraining completed!")

        if args.save_model:
            # Save training curves
            import matplotlib.pyplot as plt
            plt.savefig(Path(args.save_dir) / 'training_curves.png')

            # Save metrics
            results = {
                'final_eval_reward': master.evaluate(num_episodes=100),
                'training_rewards': master.rewards_history,
                'training_steps': master.steps_history
            }
            import json
            with open(Path(args.save_dir) / 'training_results.json', 'w') as f:
                json.dump(results, f)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
