import subprocess
import sys
import os

def install_requirements():
    print("Installing required packages...")
    
    # Install basic requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install FEniCS
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fenics-dolfin"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mshr"])
    except subprocess.CalledProcessError:
        print("Error installing FEniCS. Please install it manually:")
        print("For macOS: brew install fenics")
        print("For Ubuntu: sudo apt-get install fenics")
        print("For Windows: Download from https://fenicsproject.org/download/")
    
    print("\nInstallation complete. Please run plot_solutions.py to test the setup.")

if __name__ == "__main__":
    install_requirements() 