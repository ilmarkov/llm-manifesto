"""Build the per-pod shell script that prepares the environment and starts vLLM."""

from __future__ import annotations

import json
import shlex
from typing import Any

from .dp_ports import RolePorts
from .parallelism import parallel_layout
from .spec import DeploymentSpec, DpLoadBalancing, RoleSpec


def _flag_name(name: str) -> str:
    if "." in name:
        return "--" + name
    return "--" + name.replace("_", "-")


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _format_arg(name: str, value: Any) -> list[str]:
    flag = _flag_name(name)
    if "." in name:
        return [f"{flag}={shlex.quote(_format_value(value))}"]
    if isinstance(value, bool):
        return [flag] if value else []
    if isinstance(value, (dict, list)):
        return [flag, shlex.quote(_format_value(value))]
    return [flag, shlex.quote(str(value))]


def _command_lines(parts: list[str | list[str]], *, indent: str = "") -> list[str]:
    if not parts:
        return []
    rendered = [" ".join(part) if isinstance(part, list) else part for part in parts]
    lines = [f"{indent}{rendered[0]} \\"]
    lines.extend(f"{indent}  {part} \\" for part in rendered[1:-1])
    lines.append(f"{indent}  {rendered[-1]}")
    return lines


def build_launch_script(
    spec: DeploymentSpec,
    role: RoleSpec,
    ports: RolePorts,
    *,
    log_dir: str,
    dev_source: str,
    vllm_args: dict[str, Any] | None = None,
) -> str:
    layout = parallel_layout(role)
    external_dp = role.parallelism.dp_enabled and role.dp_load_balancing == DpLoadBalancing.EXTERNAL
    lines = [
        "set -euo pipefail",
        f"LOG_DIR={shlex.quote(log_dir)}",
        'mkdir -p "$LOG_DIR"',
        'LOG_FILE="$LOG_DIR/${HOSTNAME}_$(date +%Y%m%d-%H%M%S).log"',
        'exec > >(tee -a "$LOG_FILE") 2>&1',
        'echo "=== Pod $HOSTNAME started at $(date -Iseconds) ==="',
        "",
        f"find {shlex.quote(dev_source + '/vllm')} -name __pycache__ -type d -exec rm -rf {{}} + 2>/dev/null || true",
        'if [ -n "${MANIFESTO_VLLM_DEV_VENV:-}" ] && [ -d "${MANIFESTO_VLLM_DEV_VENV}" ]; then',
        '  echo "Using dev venv at ${MANIFESTO_VLLM_DEV_VENV}"',
        '  source "${MANIFESTO_VLLM_DEV_VENV}/bin/activate"',
        "elif [ -f /opt/vllm/bin/activate ]; then",
        "  source /opt/vllm/bin/activate",
        "fi",
        "",
    ]
    hooks = [*spec.runtime.pre_launch, *role.pre_launch]
    if hooks:
        lines += [
            "echo '=== Running pre-launch hooks ==='",
            *hooks,
            "",
        ]

    if role.parallelism.dp_enabled:
        lines += [
            f"DP_SIZE_LOCAL={layout.dp_local_size}",
            f"DP_SIZE={layout.dp_world_size}",
        ]
        if role.lws.size > 1:
            lines.append("START_RANK=$(( LWS_WORKER_INDEX * DP_SIZE_LOCAL ))")
        else:
            lines.append("START_RANK=0")
    else:
        lines += ["DP_SIZE_LOCAL=1", "START_RANK=0"]

    base_args: list[str | list[str]] = [
        "vllm",
        "serve",
        shlex.quote(spec.model.id),
        ["--port", str(ports.backend[0]) if external_dp else "$PORT"],
        ["--tensor-parallel-size", str(layout.tp_world_size)],
    ]
    if not external_dp:
        base_args[3:3] = [["--device-ids", "$GPUS"]]
    if role.parallelism.ep:
        base_args.append("--enable-expert-parallel")
    if external_dp:
        dp_address = "${LWS_LEADER_ADDRESS}" if role.lws.size > 1 else "127.0.0.1"
        base_args += [
            ["--data-parallel-size", "$DP_SIZE"],
            ["--data-parallel-start-rank", "$START_RANK"],
            ["--data-parallel-size-local", "$DP_SIZE_LOCAL"],
            ["--data-parallel-address", dp_address],
            ["--data-parallel-rpc-port", "5555"],
            "--data-parallel-multi-port-external-lb",
            ["--data-parallel-supervisor-port", "8100"],
        ]
    elif role.parallelism.dp_enabled:
        dp_address = "${LWS_LEADER_ADDRESS}" if role.lws.size > 1 else "127.0.0.1"
        base_args += [
            ["--data-parallel-size", "$DP_SIZE"],
            ["--data-parallel-rank", "$RANK"],
            ["--data-parallel-size-local", "1"],
            ["--data-parallel-address", dp_address],
            ["--data-parallel-rpc-port", "5555"],
        ]
    if role.kv_transfer_config:
        base_args.append(["--kv_transfer_config", shlex.quote(json.dumps(role.kv_transfer_config, separators=(",", ":")))])
    if spec.model.served_name:
        base_args.append(["--served-model-name", shlex.quote(spec.model.served_name)])
    for name, value in (vllm_args or role.vllm_args).items():
        if arg := _format_arg(name, value):
            base_args.append(arg)

    if external_dp:
        lines += [
            "",
            f"FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${{FLASH_ATTENTION_CUTE_DSL_CACHE_DIR}}/{role.name} \\",
            f"TILELANG_CACHE_DIR=${{TILELANG_CACHE_DIR}}/{role.name} \\",
            *_command_lines(["exec", *base_args]),
        ]
        return "\n".join(lines)

    lines += [
        "",
        "for R in $(seq 0 $((DP_SIZE_LOCAL - 1))); do",
        f"  GPU_START=$((R * {layout.tp_local_size}))",
        f"  GPUS=$(seq -s, $GPU_START $((GPU_START + {layout.tp_local_size} - 1)))",
        "  RANK=$((START_RANK + R))",
        f"  PORTS=({' '.join(str(port) for port in ports.backend)})",
        "  PORT=${PORTS[$R]}",
    ]

    lines += [
        "  VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}/rank${RANK} \\",
        "  FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR}/rank${RANK} \\",
        f"  FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${{FLASH_ATTENTION_CUTE_DSL_CACHE_DIR}}/{role.name}_rank${{RANK}} \\",
        f"  TILELANG_CACHE_DIR=${{TILELANG_CACHE_DIR}}/{role.name}_rank${{RANK}} \\",
        *_command_lines([*base_args, "&"], indent="  "),
        "done",
        "",
        "wait -n",
        "kill $(jobs -p) 2>/dev/null || true",
        "exit 1",
    ]
    return "\n".join(lines)
