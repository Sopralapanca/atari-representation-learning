"""
Test script to verify the patched atariari environment works correctly.
Run this before executing your full training code.
"""

import sys
import torch

def test_imports():
    """Test that all required imports work."""
    print("=" * 60)
    print("TEST 1: Checking imports...")
    print("=" * 60)
    
    try:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        print("✓ gymnasium imported successfully")
    except ImportError as e:
        print(f"✗ gymnasium import failed: {e}")
        return False
    
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
        print("✓ stable_baselines3 vec_env imported successfully")
    except ImportError as e:
        print(f"✗ stable_baselines3 import failed: {e}")
        print("  Run: pip install stable-baselines3")
        return False
    
    try:
        from atariari.benchmark.episodes import get_episodes
        print("✓ atariari.benchmark.episodes imported successfully")
    except ImportError as e:
        print(f"✗ atariari import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from atariari.methods.encoders import NatureCNN
        print("✓ atariari.methods.encoders imported successfully")
    except ImportError as e:
        print(f"✗ atariari encoders import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All imports successful!\n")
    return True


def test_environment_creation():
    """Test that we can create an Atari environment."""
    print("=" * 60)
    print("TEST 2: Testing environment creation...")
    print("=" * 60)
    
    try:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        
        env = gym.make("PongNoFrameskip-v4")
        print(f"✓ Created PongNoFrameskip-v4 environment")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test a reset (gymnasium returns tuple)
        obs, info = env.reset()
        print(f"✓ Reset successful, observation shape: {obs.shape}")
        
        # Test a step (gymnasium returns 5 values)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(f"✓ Step successful, observation shape: {obs.shape}")
        
        env.close()
        print("\n✓ Environment creation test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_collection():
    """Test that we can collect episodes using atariari."""
    print("=" * 60)
    print("TEST 3: Testing episode collection (small test)...")
    print("=" * 60)
    
    try:
        from atariari.benchmark.episodes import get_episodes
        
        print("Collecting 1000 steps from PongNoFrameskip-v4...")
        tr_eps, val_eps = get_episodes(
            steps=5000,  # Small number for quick test
            env_name="PongNoFrameskip-v4",
            collect_mode="random_agent",
            train_mode="train_encoder",
        )
        
        print(f"✓ Collected {len(tr_eps)} training episodes")
        print(f"✓ Collected {len(val_eps)} validation episodes")
        
        if len(tr_eps) > 0:
            print(f"  First episode length: {len(tr_eps[0])}")
            print(f"  Observation shape: {tr_eps[0][0].shape}")
        
        print("\n✓ Episode collection test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Episode collection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder_creation():
    """Test that we can create an encoder."""
    print("=" * 60)
    print("TEST 4: Testing encoder creation...")
    print("=" * 60)
    
    try:
        from atariari.methods.encoders import NatureCNN
        
        class DummyArgs:
            def __init__(self):
                self.feature_size = 512
                self.end_with_relu = True
                self.encoder_type = "Nature"
                self.no_downsample = False
                self.method = "infonce"
        
        args = DummyArgs()
        observation_shape = (4, 84, 84)  # Typical Atari shape
        
        encoder = NatureCNN(observation_shape[0], args)
        print(f"✓ Created NatureCNN encoder")
        print(f"  Input channels: {observation_shape[0]}")
        print(f"  Feature size: {args.feature_size}")
        
        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = encoder.to(device)
        dummy_input = torch.randn(1, *observation_shape).to(device)
        output = encoder(dummy_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        print("\n✓ Encoder creation test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Encoder creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ATARIARI ENVIRONMENT TEST SUITE")
    print("=" * 60 + "\n")
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
    else:
        print("Note: CUDA not available, using CPU\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    
    if results[-1][1]:  # Only continue if imports work
        results.append(("Environment Creation", test_environment_creation()))
        results.append(("Episode Collection", test_episode_collection()))
        results.append(("Encoder Creation", test_encoder_creation()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run your training code.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)