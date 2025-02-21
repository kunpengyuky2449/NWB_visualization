# PowerShell script for setting up the Python environment

Write-Host "🚀 Starting project setup..." -ForegroundColor Green

# Define virtual environment name
$VENV_DIR = "venv"

# Check if virtual environment exists
if (Test-Path $VENV_DIR) {
    Write-Host "✅ Virtual environment already exists. Skipping creation." -ForegroundColor Yellow
} else {
    Write-Host "🔧 Creating virtual environment..." -ForegroundColor Cyan
    python -m venv $VENV_DIR
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Cyan
$VENV_ACTIVATE = ".\$VENV_DIR\Scripts\Activate"
if (Test-Path $VENV_ACTIVATE) {
    & $VENV_ACTIVATE
    Write-Host "✅ Virtual environment activated." -ForegroundColor Green
} else {
    Write-Host "⚠️ Activation failed! Please activate manually using '.\venv\Scripts\Activate'" -ForegroundColor Red
}

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies from requirements.txt
if (Test-Path "requirements.txt") {
    Write-Host "📦 Installing dependencies from requirements.txt..." -ForegroundColor Cyan
    pip install -r requirements.txt
} else {
    Write-Host "⚠️ requirements.txt not found! Skipping package installation." -ForegroundColor Red
}

Write-Host "✅ Setup complete! Run '.\venv\Scripts\Activate' to activate the environment." -ForegroundColor Green