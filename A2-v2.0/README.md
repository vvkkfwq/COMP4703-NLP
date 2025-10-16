# Install Instructions

If you want to set up your own staging, environment, run the commands below. You should not need to do this if you use the GPU environment. It has been initialised with everything you need. If there is a library not installed, you can install it into your base environment on the GPU server and it will be persistent.

If you have your own GPU, or want to create your own staging environment, use the commands below.

To be clear, you DO NOT HAVE TO DO THIS ON THE GPU. All of this is in the default environment you will be initialised when you login, which should be called NLPA2.. This is just the details that will allow you to reproduce it on your own computer if you wish.

```bash
conda create -n "COMP4703A2" python=3.10
conda activate COMP4703A2
conda install nmslib==2.1.1
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install rich==13.7.1 scikit-learn transformers tqdm llama-index==0.9.40
pip install protobuf FlagEmbedding peft bs4 datasets more-itertools
pip install tokenizers sentence-transformers safetensors pandas psutil pyarrow
pip install "huggingface_hub[cli]"
```

If the FlagEmbedding install fails, you can try:

```bash
pip install git+https://github.com/FlagOpen/FlagEmbedding.git
```

Or if all else fails, do you can try:

```bash
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install -e .
```

You may need to run "huggingface-cli login" if you use a gated LLM that is not already pre-cached. You will know as you will see an error telling you that you must login in order to download model "XX", where XX is the model you set in # the ranker, reranker, or rag example.

# CPU Only

It is possible to set up a local environment for coding and staging without a GPU. All of the lines above are the same, but you must install torch using:

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

instead of:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

The example code is intrumented with a Boolean `GPU = True`. Set this value to False to use a CPU only environment. Just remember to reenable it when you run your code on the GPU or you will take 10x longer to run anything. I also instrumented the code to dump out the maximum memory used when ran, and for Llama2 in the example_rag.py file, it used around 13GB. So, you might be able to barely get it to run with 16GB of RAM on a laptop, but I expect that you really want at least 32GB. The other alternative is to just try to use a 1 or 2B parameter model in your local environment, and move to the larger ones once you are read to run on the GPU.

# Using a GPU on Windows 11

Note that if you want to use a GPU on your own Windows desktop/laptop, I strongly urge you to do this in WSL as I know this will work as I have tested it at home on an RTX 5090. If your GPU does not have at least 15GB VRAM, don't waste your time -- the models will not load. Assuming that you have an NVIDIA based GPU, do the following:

```bash
wsl --install

reboot

wsl --install ubuntu

open up an ubuntu terminal window and install the nvidia drivers and dev stack.

wget -nv -O- https://lambda.ai/install-lambda-stack.sh |
I_AGREE_TO_THE_CUDNN_LICENSE=1 sh -
```

Download and install miniconda3:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

Now create your conda environment as above.

Make sure you use `export CUDA_VISIBLE_DEVICES=0` before running a job.
If you have an AMD GPU, you will need to install pytorch using the
ROCm option. I have never tested this as I do not have a discrete AMD GPU
available to test it, but it should work in theory.

Finally, here is a very simple example script called example_run.sh which
will run the 3 example files. You may need to modify these to run as there
are a number of dependencies that are hard coded.

```bash
#!/usr/bin/env bash

# Fix default conda environment to NLPA2
source /conda/etc/profile.d/conda.sh
conda activate NLPA2
export PATH=/conda/envs/NLPA2/bin:$PATH

PYTHON="/conda/envs/NLPA2/bin/python"
WORKING_DIR="$HOME/A2-v2.0"
LOGFILE="${WORKING_DIR}/run.log"

if [ -f ${LOGFILE} ];
then
  OLD=$(mktemp -d ${LOGFILE}.XXXX)
  mv ${LOGFILE} ${OLD}
fi

echo "Running example retrieval"
${PYTHON} example_retrieval.py 2>&1 >> ${LOGFILE} 2>&1
echo "Running example reranker"
${PYTHON} example_reranker.py 2>&1 >> ${LOGFILE} 2>&1
echo "Running example rag"
${PYTHON} example_rag.py 2>&1 >> ${LOGFILE} 2>&1
```

This is only a simple example. You can split it into 3 scripts or whatever suits you. The script will only work in a Unix like environment such as WSL, MacOS, or, Linux.
