# Applying agentic coding to ARC AGI 2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![ARC-AGI](https://img.shields.io/badge/Task-ARC--AGI-red)](https://arcprize.org/)

This repository allows reproduction of the blog post [Applying agentic coding to ARC AGI 2](https://pivotools.github.io/pivotools-quarto-blog/posts/agentic_coding_arc_agi/).

## 1. Installing
```bash
pip install -e .
```

## 2. Setting up API Keys
Create an env file (e.g. `.env`) in the root directory and add the API keys (only for the model providers you will use for the solver).
```bash
OPENAI_API_KEY=your_openai_api_key
VLLM_API_KEY=your_vllm_api_key
MINIMAX_API_KEY=your_minimax_api_key
```
If using GPT-OSS 120B-High, you will need to use our patched image to start the VLLM server. See [Setting up VLLM server](#setting-up-vllm-server) for more details.

## 3. Setting up the IPython based REPL
Running our arc agi solver requires a stateful IPython based REPL to execute Python code, except for the plain COT solver, which uses reasoning models without any code interpreter.

There are two options for setting up the code interpreter:
1. Run code locally using [ipybox](https://github.com/gradion-ai/ipybox). This doesn't incur any additional cost.
2. Run code on a remote server using [daytona](https://github.com/gradion-ai/daytona). This needs an user account on their platform. 

You can choose between the two options by selecting the [solver config you are using](./src/arcagi2/solver/config/__init__.py) appropriately. The configs ending with `_daytona` use Daytona, the ones ending with `_baseline` don't use any code interpreter, and all the others use IPyBox.

### Option 1: Use IPyBox (code will be executed locally)
Build the ipybox docker image.
```bash
python -m ipybox build -t ipybox:solver -d ./src/arcagi2/utils/config/ipybox_dependencies_solver.txt
```

### Option 2: Use Daytona (code will be executed on a remote server)
Set the env var `DAYTONA_API_KEY` in your env file.

## Run the evaluation

Evaluations can be run using the command `arcagi2-evaluate`. Use `arcagi2-evaluate --help` to see all options and their meanings.

### Example command for GPT OSS 120B High using IPyBox

```bash
arcagi2-evaluate --challenge_file ./data/arc-agi_evaluation_challenges.json -c gpt_oss_120b_high -o <output_folder> -b "http://<vllm_ip_addr>:<vllm_port>/v1" -s <submission_folder> -n 2 -p <num_parallel_workers> -e <path_to_env_file> -t <timeout_hours>
```

Running this produces: 
- `submissions.json` file under `<submission_folder>`. You will need the path to this file for scoring.
- detailed traces, artifacts and logs are stored in `<output_folder`>.

## Score the submission

```bash
arcagi2-score --solutions_file ./data/arc-agi_evaluation_solutions.json --submissions_file <path_to_submissions_json>
```

## Setting up VLLM server
A patch needs to be applied to VLLM in order to use GPT-OSS 120B with reasoning effort set to `high`. You can inspect the patch by looking in the [vllm_patch folder](./src/vllm_patch). We provide the patched docker image in the GitHub Container Registry (tag `ghcr.io/gutfeeling/vllm-openai:v0.11.0-patch`).
A GitHub Actions [workflow](.github/workflows/build_and_push_patched_vllm.yml) in this repo builds and pushes the patched image to the GitHub Container Registry. 

You can start a VLLM server with the patched image using:
```bash
docker run -it --gpus all -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host ghcr.io/gutfeeling/vllm-openai:v0.11.0-patch --model openai/gpt-oss-120b --gpu_memory_utilization 0.95 --async-scheduling --tool-call-parser openai --enable-auto-tool-choice --api-key <your_vllm_api_key>
```
