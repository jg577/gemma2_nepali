echo "Setting up a script to set up the environment"
sudo apt install build-essential
pip install --upgrade torch torchvision torchaudio
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install --upgrade transformers
git clone https://github.com/LeonidasY/fast-vocabulary-transfer.git
git clone https://github.com/chapainaashish/nepali-ukhaan.git