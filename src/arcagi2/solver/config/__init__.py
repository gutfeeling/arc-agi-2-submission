from arcagi2.solver.config.claude_opus_4_5_64k import CLAUDE_OPUS_4_5_64K_AGENTIC_CODING_CONFIG, CLAUDE_OPUS_4_5_64K_DAYTONA_AGENTIC_CODING_CONFIG
from arcagi2.solver.config.claude_opus_4_5_64k_baseline import CLAUDE_OPUS_4_5_64K_PLAIN_COT_CONFIG
from arcagi2.solver.config.gpt_5_2_xhigh import GPT_5_2_XHIGH_AGENTIC_CODING_CONFIG, GPT_5_2_XHIGH_DAYTONA_AGENTIC_CODING_CONFIG
from arcagi2.solver.config.gpt_5_2_xhigh_baseline import GPT_5_2_XHIGH_PLAIN_COT_CONFIG
from arcagi2.solver.config.gpt_oss_120b_high import GPT_OSS_120B_HIGH_AGENTIC_CODING_CONFIG, GPT_OSS_120B_HIGH_DAYTONA_AGENTIC_CODING_CONFIG
from arcagi2.solver.config.gpt_oss_120b_high_baseline import GPT_OSS_120B_HIGH_PLAIN_COT_CONFIG
from arcagi2.solver.config.kimi_k2_thinking import KIMI_K2_THINKING_AGENTIC_CODING_CONFIG, KIMI_K2_THINKING_DAYTONA_AGENTIC_CODING_CONFIG
from arcagi2.solver.config.kimi_k2_thinking_baseline import KIMI_K2_THINKING_PLAIN_COT_CONFIG
from arcagi2.solver.config.minimax_m2_1 import MINIMAX_M2_1_AGENTIC_CODING_CONFIG, MINIMAX_M2_1_DAYTONA_AGENTIC_CODING_CONFIG
from arcagi2.solver.config.minimax_m2_1_baseline import MINIMAX_M2_1_PLAIN_COT_CONFIG

SOLVER_CONFIGS = {
    "claude_opus_4_5_64k": CLAUDE_OPUS_4_5_64K_AGENTIC_CODING_CONFIG,
    "claude_opus_4_5_64k_daytona": CLAUDE_OPUS_4_5_64K_DAYTONA_AGENTIC_CODING_CONFIG,
    "claude_opus_4_5_64k_baseline": CLAUDE_OPUS_4_5_64K_PLAIN_COT_CONFIG,
    "gpt_5_2_xhigh": GPT_5_2_XHIGH_AGENTIC_CODING_CONFIG,
    "gpt_5_2_xhigh_daytona": GPT_5_2_XHIGH_DAYTONA_AGENTIC_CODING_CONFIG,
    "gpt_5_2_xhigh_baseline": GPT_5_2_XHIGH_PLAIN_COT_CONFIG,
    "gpt_oss_120b_high": GPT_OSS_120B_HIGH_AGENTIC_CODING_CONFIG,
    "gpt_oss_120b_high_daytona": GPT_OSS_120B_HIGH_DAYTONA_AGENTIC_CODING_CONFIG,
    "gpt_oss_120b_high_baseline": GPT_OSS_120B_HIGH_PLAIN_COT_CONFIG,
    "kimi_k2_thinking": KIMI_K2_THINKING_AGENTIC_CODING_CONFIG,
    "kimi_k2_thinking_daytona": KIMI_K2_THINKING_DAYTONA_AGENTIC_CODING_CONFIG,
    "kimi_k2_thinking_baseline": KIMI_K2_THINKING_PLAIN_COT_CONFIG,
    "minimax_m2_1": MINIMAX_M2_1_AGENTIC_CODING_CONFIG,
    "minimax_m2_1_daytona": MINIMAX_M2_1_DAYTONA_AGENTIC_CODING_CONFIG,
    "minimax_m2_1_baseline": MINIMAX_M2_1_PLAIN_COT_CONFIG,
}