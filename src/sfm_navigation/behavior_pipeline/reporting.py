"""Unified reporting & logging for the pipeline."""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
import pandas as pd

class PipelineLogger:
    """Manages log files and per-stage reports."""

    def __init__(self, output_dir: str, log_level: str = "INFO", run_id: Optional[str] = None):
        self.output_dir = output_dir
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(output_dir, self.run_id, "logs")
        self.report_dir = os.path.join(output_dir, self.run_id, "reports")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

        self.logger = logging.getLogger(f"pipeline_{self.run_id}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, "pipeline.log"))
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.errors = []

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
        self.errors.append(msg)

    def write_stage_report(self, stage_name: str, content: str):
        path = os.path.join(self.report_dir, f"{stage_name}_report.txt")
        with open(path, 'w') as f:
            f.write(content)
        self.info(f"Report saved: {path}")

    def write_dataframe_summary(self, df: pd.DataFrame, title: str) -> str:
        buf = f"--- {title} ---\n"
        buf += f"Shape: {df.shape}\n"
        buf += df.describe(include='all').to_string() + "\n"
        buf += f"Missing values:\n{df.isnull().sum().to_string()}\n"
        return buf

    def final_summary(self):
        summary = f"Pipeline run {self.run_id} completed.\n"
        if self.errors:
            summary += f"\nErrors ({len(self.errors)}):\n"
            for e in self.errors:
                summary += f"  - {e}\n"
        else:
            summary += "No errors.\n"
        path = os.path.join(self.report_dir, "pipeline_summary.txt")
        with open(path, 'w') as f:
            f.write(summary)
        self.info(f"Final summary: {path}")