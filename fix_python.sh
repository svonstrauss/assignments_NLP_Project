#!/bin/bash

# Fix Python 3.13 setup script
echo "===== Python 3.13 Setup and Fix Script ====="
echo "This script will:"
echo "1. Install essential data science packages for Python 3.13"
echo "2. Configure your shell to prioritize Python 3.13"
echo "3. Create a launch script for Jupyter with Python 3.13"
echo ""

# Install required packages for Python 3.13
echo "Installing required packages for Python 3.13..."
/opt/homebrew/bin/python3.13 -m pip install --upgrade pip
/opt/homebrew/bin/python3.13 -m pip install pandas numpy matplotlib seaborn scikit-learn nltk jupyter notebook ipykernel

# Register Python 3.13 as a Jupyter kernel
echo "Registering Python 3.13 as a Jupyter kernel..."
/opt/homebrew/bin/python3.13 -m ipykernel install --user --name python3.13 --display-name "Python 3.13"

# Create a .zshrc.python file that can be sourced
echo "Creating Python configuration file..."
cat > ~/.zshrc.python << 'EOF'
# Python 3.13 configuration
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"

# Alias for Python 3.13
alias python="/opt/homebrew/bin/python3.13"
alias python3="/opt/homebrew/bin/python3.13"
alias pip="/opt/homebrew/bin/python3.13 -m pip"
alias pip3="/opt/homebrew/bin/python3.13 -m pip"
alias jupyter="/opt/homebrew/bin/python3.13 -m jupyter"

# Temporarily disable conda auto-activation if present
if [[ -n $CONDA_PREFIX ]]; then
    conda deactivate
fi
EOF

# Create a launcher script for Jupyter with Python 3.13
echo "Creating a Jupyter launcher for your project..."
cat > ~/Desktop/Class\ NLP\ Class\ Tulane/assignments_NLP_Project/launch_jupyter.sh << 'EOF'
#!/bin/bash

# Source Python 3.13 configuration
source ~/.zshrc.python

# Navigate to project directory
cd ~/Desktop/Class\ NLP\ Class\ Tulane/assignments_NLP_Project

# Launch Jupyter notebook with Python 3.13
/opt/homebrew/bin/python3.13 -m jupyter notebook
EOF

# Make the launcher executable
chmod +x ~/Desktop/Class\ NLP\ Class\ Tulane/assignments_NLP_Project/launch_jupyter.sh

# Create a script to use Python 3.13
cat > ~/Desktop/Class\ NLP\ Class\ Tulane/assignments_NLP_Project/use_python3.13.sh << 'EOF'
#!/bin/bash

# Source Python 3.13 configuration
source ~/.zshrc.python

# Launch a new shell with Python 3.13 as default
exec $SHELL
EOF

# Make the script executable
chmod +x ~/Desktop/Class\ NLP\ Class\ Tulane/assignments_NLP_Project/use_python3.13.sh

echo ""
echo "===== Setup Complete! ====="
echo ""
echo "To use Python 3.13 for your project:"
echo ""
echo "1. Launch Jupyter with Python 3.13:"
echo "   ./launch_jupyter.sh"
echo ""
echo "2. Open a terminal with Python 3.13 as default:"
echo "   ./use_python3.13.sh"
echo ""
echo "3. In Jupyter, select the 'Python 3.13' kernel from the Kernel menu"
echo ""
echo "===== Testing Python 3.13 Installation ====="
/opt/homebrew/bin/python3.13 -c "import pandas; print(f'Pandas {pandas.__version__} is successfully installed!')" || echo "Error importing pandas"