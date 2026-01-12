from arcagi2.solver.config.claude_opus_4_5_64k import CLAUDE_OPUS_4_5_64K_SYSTEM_CONFIG, CLAUDE_OPUS_4_5_64K_DAYTONA_SYSTEM_CONFIG
from arcagi2.solver.config.claude_opus_4_5_64k_baseline import CLAUDE_OPUS_4_5_64K_BASELINE_CONFIG
from arcagi2.solver.config.gpt_5_2_xhigh import GPT_5_2_XHIGH_SYSTEM_CONFIG, GPT_5_2_XHIGH_DAYTONA_SYSTEM_CONFIG
from arcagi2.solver.config.gpt_5_2_xhigh_baseline import GPT_5_2_XHIGH_BASELINE_CONFIG
from arcagi2.solver.config.gpt_oss_120b_high import GPT_OSS_120B_HIGH_SYSTEM_CONFIG, GPT_OSS_120B_HIGH_DAYTONA_SYSTEM_CONFIG
from arcagi2.solver.config.gpt_oss_120b_high_baseline import GPT_OSS_120B_HIGH_BASELINE_CONFIG
from arcagi2.solver.config.minimax_m2_1 import MINIMAX_M2_1_SYSTEM_CONFIG, MINIMAX_M2_1_DAYTONA_SYSTEM_CONFIG
from arcagi2.solver.config.minimax_m2_1_baseline import MINIMAX_M2_1_BASELINE_CONFIG

SOLVER_CONFIGS = {
    "claude_opus_4_5_64k": CLAUDE_OPUS_4_5_64K_SYSTEM_CONFIG,
    "claude_opus_4_5_64k_daytona": CLAUDE_OPUS_4_5_64K_DAYTONA_SYSTEM_CONFIG,
    "claude_opus_4_5_64k_baseline": CLAUDE_OPUS_4_5_64K_BASELINE_CONFIG,
    "gpt_5_2_xhigh": GPT_5_2_XHIGH_SYSTEM_CONFIG,
    "gpt_5_2_xhigh_baseline": GPT_5_2_XHIGH_BASELINE_CONFIG,
    "gpt_5_2_xhigh_daytona": GPT_5_2_XHIGH_DAYTONA_SYSTEM_CONFIG,
    "gpt_oss_120b_high": GPT_OSS_120B_HIGH_SYSTEM_CONFIG,
    "gpt_oss_120b_high_baseline": GPT_OSS_120B_HIGH_BASELINE_CONFIG,
    "gpt_oss_120b_high_daytona": GPT_OSS_120B_HIGH_DAYTONA_SYSTEM_CONFIG,
    "minimax_m2_1": MINIMAX_M2_1_SYSTEM_CONFIG,
    "minimax_m2_1_daytona": MINIMAX_M2_1_DAYTONA_SYSTEM_CONFIG,
    "minimax_m2_1_baseline": MINIMAX_M2_1_BASELINE_CONFIG,
}