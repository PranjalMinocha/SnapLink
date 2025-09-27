# 🚀 Project SnapLink

*A brief, one-sentence description of what your project does.*

This guide provides all the necessary steps to set up the development environment and get the project running. This project uses a standard `src` layout for clean, maintainable code.

---

## ⚙️ Getting Started: Environment Setup

Follow these instructions carefully to ensure a consistent development environment across the team.

**Core Technologies:**
*   **Language:** Python 3.12.4
*   **Environment Manager:** `venv`
*   **Version Manager (macOS):** `pyenv`

---

### For macOS Users

#### Step 1: Install Homebrew
Homebrew is the package manager for macOS. If you don't have it, open your terminal and run:
```bash
/bin/bash -c "\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Step 2: Install and Configure `pyenv`
`pyenv` allows us to use the exact Python version required for this project.

1.  **Install `pyenv` via Homebrew:**
    ```bash
    brew install pyenv
    ```

2.  **Configure Your Shell:** This step is critical for `pyenv` to work correctly. Add the following block to the **very end** of your `~/.zshrc` file:
    ```bash
    # --- Pyenv Configuration ---
    export PYENV_ROOT="\$HOME/.pyenv"
    command -v pyenv >/dev/null || export PATH="\$PYENV_ROOT/bin:\$PATH"
    eval "\$(pyenv init -)"
    # --- End of Pyenv Configuration ---
    ```

3.  **IMPORTANT:** **Restart your terminal.** (Completely quit with `Cmd + Q` and open a new one).

#### Step 3: Install Python 3.12.4
Use `pyenv` to install our project's specific Python version.
```bash
pyenv install 3.12.4
```

#### Step 4: Set the Global Python Version
This command tells your system to use Python 3.12.4 by default.
```bash
pyenv global 3.12.4
```

#### Step 5: Verify Your Setup
Restart your terminal one last time. Run the following command to confirm the version:
```bash
python --version
# Expected Output: Python 3.12.4
```

---

### For Windows Users

#### Step 1: Install Python 3.12.4
1.  Go to the official Python downloads page: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2.  Download the **"Windows installer (64-bit)"** for **Python 3.12.4**.
3.  Run the installer. **CRITICALLY, on the first screen, you must check the box that says "Add python.exe to PATH"**.

#### Step 2: Verify Your Setup
Open a new Command Prompt (`cmd`) or PowerShell window and run:
```bash
python --version
# Expected Output: Python 3.12.4
```

---

## 💻 Project Installation (For Everyone)

Once your system has Python 3.12.4 installed and verified, follow these steps.

#### Step 1: Clone the Repository
Clone the project to your local machine using the SSH option.
```bash
git clone git@github.com:PranjalMinocha/SnapLink.git
cd SnapLink
```

#### Step 2: Create and Activate the Virtual Environment
This creates an isolated environment for our project's packages. We use the custom name `tsl`.

1.  **Create the environment:**
    ```bash
    python -m venv tsl
    ```
2.  **Activate the environment:**
    *   **On macOS/Linux:**
        ```bash
        source tsl/bin/activate
        ```
    *   **On Windows:**
        ```bash
        .\\tsl\\Scripts\\activate
        ```
    Your terminal prompt will now start with `(tsl)`.

#### Step 3: **CRITICAL** - Install the Project in Editable Mode
Because we use a `src` layout, this command makes our application code findable. **Run this from the project's root directory.**
```bash
pip install -e .
```

#### Step 4: Install Dependencies
This command installs all third-party libraries listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

---

You're all set. Happy hacking!