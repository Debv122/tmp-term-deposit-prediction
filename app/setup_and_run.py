import subprocess
import sys
import os

def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)  # Go up one level from app/ to project root

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    project_root = get_project_root()
    requirements_path = os.path.join(project_root, "requirements.txt")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("Error installing requirements. Please install manually:")
        print("pip install pandas numpy matplotlib seaborn scikit-learn")

def check_data_files():
    """Check if data files exist"""
    project_root = get_project_root()
    data_files = [
        'data/data/bank-additional-full.csv',
        'data/data/bank-additional.csv',
        'data/data/bank-full.csv',
        'data/data/bank.csv'
    ]
    
    print("Checking data files...")
    for file in data_files:
        full_path = os.path.join(project_root, file)
        if os.path.exists(full_path):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} not found")
    
    main_data_path = os.path.join(project_root, 'data/data/bank-additional-full.csv')
    return os.path.exists(main_data_path)

def main():
    print("Setting up Term Deposit Prediction Project...")
    print("="*50)
    
    # Install requirements
    install_requirements()
    
    # Check data files
    if check_data_files():
        print("\nData files found! Ready to run analysis.")
        print("\nTo run the analysis, execute:")
        print("python app/term_deposit_prediction.py")
    else:
        print("\nWarning: Main data file not found!")
        print("Please ensure data/data/bank-additional-full.csv exists")

if __name__ == "__main__":
    main()
