set -e
if [[ "$(uname)" == "Darwin" ]]; then
    # Running on macOS (Darwin)
    echo "Running on macOS. Attempting to upgrade via Homebrew..."
    brew upgrade gemini-cli
else
    # Running on Linux or other OS
    echo "Running on Linux/Other OS. Installing/upgrading via global npm..."
    rm -rf /usr/local/share/nvm/versions/node/*/lib/node_modules/@google/gemini-cli
    npm install -g @google/gemini-cli@latest
fi