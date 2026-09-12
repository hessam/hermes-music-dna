"""Process Supervisor: Runs the Telegram Bot and Pipeline Worker concurrently with auto-restart and persistent logging."""
import asyncio
import logging
import os
import sys

LOG_FILE = "/workspace/jobs/agent.log"
os.makedirs("/workspace/jobs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Supervisor] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("supervisor")


async def run_process(name: str, cmd: list[str]) -> None:
    log_fp = open(LOG_FILE, "a", buffering=1, encoding="utf-8")
    while True:
        logger.info("Starting %s: %s", name, " ".join(cmd))
        env = dict(os.environ)
        env["PYTHONPATH"] = "/workspace/workspace"
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=log_fp,
            stderr=log_fp,
            env=env,
            cwd="/workspace/workspace/music_dna_agent",
        )
        ret_code = await proc.wait()
        logger.warning("%s process exited with code %s. Restarting in 3 seconds...", name, ret_code)
        await asyncio.sleep(3)


async def main() -> None:
    logger.info("🚀 Launching Music DNA Agent Supervisor (Logging to %s)...", LOG_FILE)
    tasks = [
        asyncio.create_task(run_process("Pipeline Worker", [sys.executable, "-m", "music_dna_agent.src.worker.pipeline_worker"])),
        asyncio.create_task(run_process("Telegram Bot", [sys.executable, "-m", "music_dna_agent.src.bot"])),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
