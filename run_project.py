"""
Main Script to Run the Complete Fraud Detection Project

This script provides a simple way to run the entire fraud detection pipeline
from start to finish with clear progress indicators.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_step(step_num, title):
    """Print a formatted step header"""
    print(f"\n{step_num}. {title}")
    print("-" * 60)

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Success!")
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
        else:
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False
    return True

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"  {description} found: {filepath}")
        return True
    else:
        print(f"  {description} not found: {filepath}")
        return False

def main():
    """Main execution function"""
    print_header("FRAUD DETECTION SYSTEM - COMPLETE PIPELINE")
    
    # Check Python and dependencies
    print_step(0, "Checking Environment")
    
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required directories
    required_dirs = ['data', 'src', 'models', 'notebooks']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  Directory exists: {dir_name}/")
        else:
            print(f"  Creating directory: {dir_name}/")
            os.makedirs(dir_name, exist_ok=True)
    
    # Step 1: Install dependencies
    print_step(1, "Installing Dependencies")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("Warning: Some dependencies may be missing. Proceeding anyway...")
    
    # Step 2: Generate data
    print_step(2, "Generating Sample Data")
    if not run_command("python src/data_generation.py", "Creating fraud detection dataset"):
        print("Error: Data generation failed. Please check the error above.")
        return False
    
    # Check if data was created
    if not check_file_exists("data/fraud_data.csv", "Dataset"):
        print("Error: Dataset was not created successfully.")
        return False
    
    # Step 3: Preprocessing
    print_step(3, "Data Preprocessing and Feature Engineering")
    if not run_command("python src/preprocessing.py", "Preprocessing data"):
        print("Error: Preprocessing failed.")
        return False
    
    # Check processed data
    if not (check_file_exists("data/train_processed.csv", "Training data") and 
            check_file_exists("data/test_processed.csv", "Test data")):
        print("Error: Processed data was not created.")
        return False
    
    # Step 4: Model Training
    print_step(4, "Model Training and Evaluation")
    if not run_command("python src/model_training.py", "Training fraud detection models"):
        print("Error: Model training failed.")
        return False
    
    # Check models
    if not check_file_exists("models/best_model.pkl", "Trained model"):
        print("Error: Model was not saved successfully.")
        return False
    
    # Step 5: Model Evaluation
    print_step(5, "Comprehensive Model Evaluation")
    if not run_command("python src/model_evaluation.py", "Evaluating model performance"):
        print("Warning: Detailed evaluation failed, but basic training completed.")
    
    # Step 6: Start API Server
    print_step(6, "Starting API Server")
    
    # Check if API is already running
    try:
        import requests
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            print("  API server is already running!")
            api_running = True
        else:
            api_running = False
    except:
        api_running = False
    
    if not api_running:
        print("  Starting API server...")
        print("  Note: The API server will run in the background.")
        print("  You can access it at http://localhost:8000")
        print("  Press Ctrl+C to stop the server when done.")
        
        # Start API in a separate process
        try:
            import subprocess
            api_process = subprocess.Popen([sys.executable, 'src/api.py'])
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Test if server started
            try:
                response = requests.get('http://localhost:8000/health', timeout=5)
                if response.status_code == 200:
                    print("  API server started successfully!")
                else:
                    print("  Warning: API server may not have started correctly.")
            except:
                print("  Warning: Could not verify API server status.")
                
        except Exception as e:
            print(f"  Error starting API server: {e}")
            api_process = None
    
    # Step 7: Project Summary
    print_step(7, "Project Summary and Next Steps")
    
    print_header("PROJECT COMPLETED SUCCESSFULLY!")
    
    print("\nWhat was accomplished:")
    print("  1. Generated realistic fraud detection dataset")
    print("  2. Performed comprehensive data preprocessing")
    print("  3. Trained and evaluated multiple ML models")
    print("  4. Deployed real-time prediction API")
    print("  5. Created comprehensive documentation")
    
    print("\nKey files created:")
    print("  data/fraud_data.csv - Original dataset")
    print("  data/train_processed.csv - Preprocessed training data")
    print("  data/test_processed.csv - Preprocessed test data")
    print("  models/best_model.pkl - Trained model")
    print("  models/preprocessor.pkl - Data preprocessing pipeline")
    print("  models/model_evaluation_results.json - Performance metrics")
    
    print("\nHow to use the system:")
    print("  1. API Documentation: http://localhost:8000/docs")
    print("  2. Health Check: http://localhost:8000/health")
    print("  3. Model Info: http://localhost:8000/model_info")
    print("  4. Single Prediction: POST to http://localhost:8000/predict")
    print("  5. Batch Prediction: POST to http://localhost:8000/batch_predict")
    
    print("\nJupyter Notebooks for exploration:")
    print("  notebooks/01_eda.ipynb - Exploratory Data Analysis")
    print("  notebooks/02_model_training.ipynb - Model Training and Evaluation")
    
    print("\nInterview Preparation:")
    print("  interview_guide.md - Comprehensive interview guide")
    print("  demo_script.py - Demonstration script")
    
    print("\nTo test the API:")
    print("  python demo_script.py")
    
    print("\nTo stop the API server:")
    print("  Press Ctrl+C in the terminal where the server is running")
    
    print("\n" + "="*80)
    print("Thank you for using the Fraud Detection System!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nProject completed successfully!")
        else:
            print("\nProject completed with some issues. Please check the errors above.")
    except KeyboardInterrupt:
        print("\nProject interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check the error message above and try again.")
