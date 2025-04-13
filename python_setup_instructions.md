# Python Setup Instructions

## Current Python Installation Status

You currently have multiple Python installations:
- Anaconda Python 3.12.2 (default in PATH)
- Homebrew Python 3.13.2
- Homebrew Python 3.12.9
- PyEnv Python 3.11.11
- System Python installations

## Recommended Steps to Clean Up Python Installation

### 1. Create a Project-Specific Virtual Environment

```bash
# Navigate to your project directory
cd ~/Desktop/Class\ NLP\ Class\ Tulane/assignments_NLP_Project

# Create a virtual environment using Homebrew's Python 3.13
/opt/homebrew/bin/python3.13 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn nltk newspaper3k jupyter
```

### 2. Update Your Shell Profile (.zshrc or .bash_profile)

Add these lines to your shell profile to organize your Python installation:

```bash
# Python path management
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"

# Alias for different Python versions
alias python-conda="/opt/anaconda3/bin/python"
alias python3-conda="/opt/anaconda3/bin/python3"
alias python3-brew="/opt/homebrew/bin/python3"
alias python3.13="/opt/homebrew/bin/python3.13"
alias python3.12="/opt/homebrew/bin/python3.12"

# Function to list available Python versions
python-versions() {
  echo "Available Python versions:"
  echo "  1. System Python: $(python3 --version 2>/dev/null || echo 'Not found')"
  echo "  2. Homebrew Python 3.13: $(/opt/homebrew/bin/python3.13 --version 2>/dev/null || echo 'Not found')"
  echo "  3. Homebrew Python 3.12: $(/opt/homebrew/bin/python3.12 --version 2>/dev/null || echo 'Not found')"
  echo "  4. Anaconda Python: $(/opt/anaconda3/bin/python --version 2>/dev/null || echo 'Not found')"
  echo "  5. PyEnv Python: $(~/.pyenv/versions/3.11.11/bin/python --version 2>/dev/null || echo 'Not found')"
  echo ""
  echo "Current active Python: $(which python) ($(python --version 2>&1))"
}
```

### 3. Optional: Set Homebrew Python as Default

If you want to make Homebrew's Python 3.13.2 your default Python:

```bash
# Add to your shell profile
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
```

### 4. Clean Up Unused Python Installations (Optional)

If you want to remove Anaconda completely (after backing up any important environments):

```bash
# Remove Anaconda
rm -rf /opt/anaconda3

# Remove Anaconda from PATH in your shell profile
# Edit .zshrc or .bash_profile and remove any Anaconda references
```

### 5. Project-Specific Instructions

When working on this project:

1. Always activate the virtual environment:
   ```bash
   cd ~/Desktop/Class\ NLP\ Class\ Tulane/assignments_NLP_Project
   source venv/bin/activate
   ```

2. Run your Python scripts through the virtual environment:
   ```bash
   python project/wordCount.py
   ```

3. Launch Jupyter notebooks:
   ```bash
   jupyter notebook project/report/milestone_report.ipynb
   ```

This setup isolates your project dependencies while maintaining access to different Python versions when needed.