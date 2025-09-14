#!/bin/bash

# A script to set up the Python environment and run the LLM Proxy.

# --- Configuration ---
VENV_DIR=".venv"
PYTHON_CMD="python3"
HOST="0.0.0.0"
PORT="8001"

# --- Colors for better output ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- Check if Python is installed ---
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo -e "${RED}ERROR: '$PYTHON_CMD' command not found.${NC}"
    echo "Please install Python 3.9+ and ensure it's in your PATH."
    exit 1
fi

# --- Set up Virtual Environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one at '$VENV_DIR'...${NC}"
    $PYTHON_CMD -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to create the virtual environment.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Existing virtual environment found.${NC}"
fi

# --- Activate Virtual Environment ---
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# --- Install Dependencies ---
echo -e "\n${YELLOW}--- Installing Dependencies ---${NC}"

echo "Installing packages from requirements.txt..."
pip install -r requirements.txt || { echo -e "${RED}ERROR: Failed to install requirements.${NC}"; exit 1; }

echo "Installing GiNZA model 'ja_ginza'..."
pip install ja_ginza || { echo -e "${RED}ERROR: Failed to install ja_ginza.${NC}"; exit 1; }

echo "Running Argos Translate setup..."
python setup_argos.py || { echo -e "${RED}ERROR: Argos Translate setup script failed.${NC}"; exit 1; }

echo -e "${GREEN}All Python dependencies installed successfully.${NC}"

# --- Check for JMdict (Critical Prerequisite) ---
echo -e "\n${YELLOW}--- Checking for JMdict file ---${NC}"
if [ ! -f "jmdict.json" ]; then
    echo -e "${RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${NC}"
    echo -e "${RED}!! CRITICAL ERROR: 'jmdict.json' not found.                !!${NC}"
    echo -e "${YELLOW}!! This file is required and must be generated manually once.  !!${NC}"
    echo -e "${YELLOW}!!                                                           !!${NC}"
    echo -e "${YELLOW}!! To fix this:                                              !!${NC}"
    echo -e "${YELLOW}!! 1. Download 'JMdict_e.gz' from the JMdict Project Page.   !!${NC}"
    echo -e "${YELLOW}!! 2. Unzip it to get the 'JMdict_e' file in this directory. !!${NC}"
    echo -e "${YELLOW}!! 3. Run this command: python parse_jmdict.py             !!${NC}"
    echo -e "${YELLOW}!! 4. Re-run this setup script.                              !!${NC}"
    echo -e "${RED}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!${NC}"
    exit 1
fi

echo -e "${GREEN}Found 'jmdict.json'. Prerequisites are met.${NC}"


# --- Launch the Application ---
echo -e "\n${GREEN}--- Setup Complete ---${NC}"
echo -e "Launching the LLM Linguistic Enhancement Proxy..."
echo -e "Access the dashboard at: ${YELLOW}http://${HOST}:${PORT}${NC}"
echo "Press CTRL+C to stop the server."

uvicorn run:app --host $HOST --port $PORT
