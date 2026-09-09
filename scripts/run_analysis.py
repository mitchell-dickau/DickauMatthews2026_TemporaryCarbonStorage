import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def main():
    """
    Executes the entire analysis pipeline and generates all figures sequentially.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Resolve paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    python_exe = sys.executable

    # Define the sequence of scripts to run
    analysis_pipeline = [
        "scripts/figure1.py",
        "scripts/figure2.py",
        "scripts/figure3.py",
        "scripts/figure4.py",
        "scripts/figure5.py",
        "scripts/figure6.py",
        "scripts/figure7.py",
        "scripts/figure8.py",
        "scripts/si_figure1_2.py",
        "scripts/si_figure3.py",
        "scripts/si_figure4.py",
    ]

    logger.info("Starting execution of the entire analysis pipeline...")

    for script in analysis_pipeline:
        script_path = os.path.join(project_root, script)
        logger.info(f"Running {script}...")

        # Run each script as a subprocess in the same environment and directory
        result = subprocess.run(
            [python_exe, script_path], cwd=project_root, check=False
        )
        if result.returncode != 0:
            logger.error(
                f"Execution failed for {script} with exit code {result.returncode}"
            )
            sys.exit(result.returncode)

    logger.info(
        "Entire analysis pipeline executed successfully! All outputs generated."
    )


if __name__ == "__main__":
    main()
