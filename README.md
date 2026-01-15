# ARC AGI Solver using LLMs (and optionally using python code interpreter)
## 1. Installing
```bash
pip install -e .
```
## 2. Setting up API Keys
Create an env file (e.g. `.env`) in the root directory and add the API keys (only for the model providers you will use for the solver).
```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
VLLM_API_KEY=your_vllm_api_key
MINIMAX_API_KEY=your_minimax_api_key
MOONSHOT_AI_API_KEY=your_moonshot_ai_api_key
```
If using GPT-OSS 120B-High, you will need to set up a self-managed VLLM server with our patch for the Responses API. See [Setting up VLLM server](#setting-up-vllm-server) for more details.
## 3. Setting up code interpreter
Running our arc agi solver requires a code interpreter to execute Python code, except for the baseline solver, which uses reasoning models without any code interpreter.

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
Set the env var `DAYTONA_API_KEY` in your env file (that you need to supply to the solver script).
## Run the solver
```bash
arcagi2-solve -c <config_name> --puzzle_json_path <puzzle_json_path> --output_folder <output_folder> -e <env_file>
```

## Setting up VLLM server
A patch needs to be applied to VLLM in order to use GPT-OSS 120B with reasoning effort set to `high`. This patch is included in the [vllm_patch folder](./src/vllm_patch). To use it, you can pull the patched docker image from the GitHub Container Registry (tag `ghcr.io/gutfeeling/vllm-openai:v0.11.0-patch`).
A GitHub Actions [workflow](.github/workflows/build_and_push_patched_vllm.yml) is included to build and push the patched image to the GitHub Container Registry. 

Afterwards, starting a VLLM server with the patched image is as simple as:
```bash
docker run -it --gpus all -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host ghcr.io/gutfeeling/vllm-openai:v0.11.0-patch --model openai/gpt-oss-120b --gpu_memory_utilization 0.95 --async-scheduling --tool-call-parser openai --enable-auto-tool-choice --api-key <your_vllm_api_key>
```