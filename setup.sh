set -e
cd ~
echo "This will need some login with your Hugging Face account. See the popups in the VSCode UI above."
echo "It uses a write access token as the password, which you can create in your Hugging Face account settings."
echo "Visit https://huggingface.co/settings/tokens to create a new token if you don't have one yet."
# Only clone if it doesn't already exist:
if [ ! -d "./gemma-3-1b-it-qat-q4_0-gguf" ]; then
  git clone https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf
fi

# Install Bazel
sudo apt-get update && sudo apt-get install -y curl gnupg clang-format && \
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg && \
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/ && \
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list && \
sudo apt-get update && sudo apt-get install -y bazel

npm install -g @google/gemini-cli

cd ~
# Remove gguf-tools if it already exists:
rm -rf gguf-tools
git clone https://github.com/antirez/gguf-tools.git
cd gguf-tools
make
sudo cp gguf-tools /usr/local/bin/