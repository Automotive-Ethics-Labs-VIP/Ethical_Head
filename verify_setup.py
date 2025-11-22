#!/usr/bin/env python3
"""
AEL Team A - Environment Verification Script
============================================
Verifies that all dependencies are correctly installed and compatible.
"""

import sys
from importlib import import_module
from packaging import version

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check_python_version():
    """Verify Python version is compatible (3.9.x or 3.10.x)."""
    print("\n" + "=" * 60)
    print("Python Version Check")
    print("=" * 60)
    
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    required_min = "3.9.0"
    required_max = "3.11.0"
    
    if version.parse(py_version) >= version.parse(required_min) and \
       version.parse(py_version) < version.parse(required_max):
        print(f"{GREEN}✓ Python {py_version} (compatible){RESET}")
        return True
    else:
        print(f"{RED}✗ Python {py_version} (requires 3.9.x or 3.10.x for CARLA){RESET}")
        return False

def check_packages():
    """Verify all required packages are installed."""
    print("\n" + "=" * 60)
    print("Package Installation Check")
    print("=" * 60)
    
    packages = [
        ("torch", "2.2.2", "PyTorch"),
        ("stable_baselines3", "2.4.0", "Stable-Baselines3"),
        ("gymnasium", "0.29.1", "Gymnasium"),
        ("numpy", "1.26.4", "NumPy"),
        ("pandas", "2.2.2", "Pandas"),
        ("scipy", "1.13.0", "SciPy"),
        ("sklearn", None, "Scikit-learn"),
        ("matplotlib", "3.8.4", "Matplotlib"),
        ("tensorboard", "2.16.2", "TensorBoard"),
        ("yaml", None, "PyYAML"),
        ("tqdm", "4.66.2", "tqdm"),
        ("hydra", "1.3.2", "Hydra"),
    ]
    
    all_passed = True
    
    for pkg_import, expected_ver, pkg_name in packages:
        try:
            module = import_module(pkg_import)
            installed_ver = getattr(module, "__version__", "unknown")
            
            if expected_ver and installed_ver != "unknown":
                if installed_ver == expected_ver:
                    print(f"{GREEN}✓ {pkg_name}: {installed_ver}{RESET}")
                else:
                    print(f"{YELLOW}~ {pkg_name}: {installed_ver} (expected {expected_ver}){RESET}")
            else:
                print(f"{GREEN}✓ {pkg_name}: {installed_ver}{RESET}")
        except ImportError:
            print(f"{RED}✗ {pkg_name}: NOT INSTALLED{RESET}")
            all_passed = False
    
    return all_passed

def check_torch_cuda():
    """Verify PyTorch CUDA/MPS availability."""
    print("\n" + "=" * 60)
    print("GPU Acceleration Check")
    print("=" * 60)
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"{GREEN}✓ CUDA version: {torch.version.cuda}{RESET}")
            print(f"{GREEN}✓ GPU count: {torch.cuda.device_count()}{RESET}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"{GREEN}✓ Apple MPS (Metal) available{RESET}")
        else:
            print(f"{YELLOW}~ No GPU acceleration available (CPU only){RESET}")
        
        return True
    except Exception as e:
        print(f"{RED}✗ Error checking GPU: {e}{RESET}")
        return False

def check_carla():
    """Check if CARLA Python API is installed."""
    print("\n" + "=" * 60)
    print("CARLA Installation Check")
    print("=" * 60)
    
    try:
        import carla
        print(f"{GREEN}✓ CARLA Python API installed{RESET}")
        # Try to get version if available
        if hasattr(carla, '__version__'):
            print(f"  Version: {carla.__version__}")
        return True
    except ImportError:
        print(f"{YELLOW}~ CARLA Python API not installed{RESET}")
        print("  Install from CARLA 0.9.15 distribution:")
        print("  pip install /path/to/carla-0.9.15-cpXX-cpXX-PLATFORM.whl")
        return False

def check_stable_baselines3():
    """Verify Stable-Baselines3 PPO is working."""
    print("\n" + "=" * 60)
    print("Stable-Baselines3 PPO Check")
    print("=" * 60)
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        import gymnasium as gym
        
        # Quick test with a simple environment
        env = gym.make("CartPole-v1")
        print(f"{GREEN}✓ PPO import successful{RESET}")
        print(f"{GREEN}✓ Gymnasium environment creation successful{RESET}")
        env.close()
        return True
    except Exception as e:
        print(f"{RED}✗ Stable-Baselines3 error: {e}{RESET}")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("  AEL Team A - Environment Verification")
    print("=" * 60)
    
    results = {
        "Python Version": check_python_version(),
        "Packages": check_packages(),
        "PyTorch GPU": check_torch_cuda(),
        "CARLA": check_carla(),
        "Stable-Baselines3": check_stable_baselines3(),
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {check_name}: {status}")
        if not passed and check_name != "CARLA":  # CARLA is optional for initial setup
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print(f"{GREEN}Environment is ready for development!{RESET}")
    else:
        print(f"{YELLOW}Some checks failed. Please review the output above.{RESET}")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
