"""Single-job VectorDBBench process manager for the local FastAPI service."""

from __future__ import annotations

import copy
import os
import shutil
import sqlite3
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

import yaml

from .index_profiles import profile_for


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
ACTIVE_STATUSES = {"queued", "running", "cancelling"}


class BenchmarkJobConflictError(RuntimeError):
    """Raised when a benchmark is already using the shared VDBBench collection."""


@dataclass(frozen=True, slots=True)
class BenchmarkJobParameters:
    command: str
    uri: str
    num_shards: int
    replica_number: int
    case_type: str
    drop_old: bool
    load: bool
    load_concurrency: int
    search_serial: bool
    search_concurrent: bool
    k: int
    concurrency_duration: int
    num_concurrency: tuple[int, ...]
    concurrency_timeout: int
    index_parameters: dict[str, Any]
    db_label: str


@dataclass(slots=True)
class BenchmarkJob:
    job_id: str
    parameters: BenchmarkJobParameters
    task_label_prefix: str
    config_path: Path
    runner_log_path: Path
    parameter_sets: tuple[BenchmarkJobParameters, ...] = field(default_factory=tuple)
    repetitions: int = 1
    total_runs: int = 1
    completed_runs: int = 0
    current_run_number: int = 0
    result_run_ids: list[str] = field(default_factory=list)
    status: str = "queued"
    phase: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    error: str | None = None
    result_run_id: str | None = None
    result_case_index: int | None = None
    cancel_requested: bool = False
    process_id: int | None = None
    started_monotonic: float | None = None


class BenchmarkJobManager:
    """Run at most one benchmark process against the shared VDBBench collection."""

    def __init__(
        self,
        project_root: Path,
        base_config_path: Path,
        runner_path: Path,
        database_path: Path,
        jobs_root: Path,
    ) -> None:
        self.project_root = project_root.resolve()
        self.base_config_path = base_config_path.resolve()
        self.runner_path = runner_path.resolve()
        self.database_path = database_path.resolve()
        self.jobs_root = jobs_root.resolve()
        self._lock = threading.RLock()
        self._jobs: dict[str, BenchmarkJob] = {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._active_job_id: str | None = None

    def _build_job_config(
        self,
        parameters: BenchmarkJobParameters,
        task_label_prefix: str,
    ) -> dict[str, Any]:
        profile_for(parameters.command)
        template_path = (
            self.base_config_path.parent / f"{parameters.command}.yml"
        )
        with template_path.open("r", encoding="utf-8") as file:
            document = yaml.safe_load(file) or {}
        if not isinstance(document, dict):
            raise ValueError("base benchmark configuration must be a YAML object")
        profile = document.get(parameters.command)
        if not isinstance(profile, dict):
            raise ValueError(
                f"base configuration does not contain {parameters.command}"
            )

        generated = copy.deepcopy(document)
        generated_profile = generated[parameters.command]
        generated_profile["uri"] = parameters.uri
        generated_profile["num_shards"] = parameters.num_shards
        generated_profile["replica_number"] = parameters.replica_number
        generated_profile["case_type"] = parameters.case_type
        generated_profile["drop_old"] = parameters.drop_old
        generated_profile["load"] = parameters.load
        generated_profile["load_concurrency"] = parameters.load_concurrency
        generated_profile["search_serial"] = parameters.search_serial
        generated_profile["search_concurrent"] = parameters.search_concurrent
        generated_profile["k"] = parameters.k
        generated_profile["concurrency_duration"] = parameters.concurrency_duration
        generated_profile["num_concurrency"] = ",".join(
            str(value) for value in parameters.num_concurrency
        )
        generated_profile["concurrency_timeout"] = parameters.concurrency_timeout
        generated_profile.update(parameters.index_parameters)
        generated_profile["db_label"] = parameters.db_label
        generated_profile["task_label"] = task_label_prefix
        return generated

    def prepare_job(
        self,
        parameters: BenchmarkJobParameters,
        job_id: str | None = None,
    ) -> BenchmarkJob:
        resolved_job_id = job_id or uuid.uuid4().hex
        task_prefix = profile_for(parameters.command)["task_prefix"]
        task_label_prefix = f"{task_prefix}-ui-{resolved_job_id[:8]}"
        job_directory = self.jobs_root / resolved_job_id
        config_path = job_directory / f"{parameters.command}.yml"
        runner_log_path = job_directory / "runner.log"
        config_document = self._build_job_config(parameters, task_label_prefix)

        job_directory.mkdir(parents=True, exist_ok=False)
        config_path.write_text(
            yaml.safe_dump(
                config_document,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return BenchmarkJob(
            job_id=resolved_job_id,
            parameters=parameters,
            task_label_prefix=task_label_prefix,
            config_path=config_path,
            runner_log_path=runner_log_path,
            parameter_sets=(parameters,),
        )

    def prepare_experiment(
        self,
        parameter_sets: tuple[BenchmarkJobParameters, ...],
        repetitions: int,
        job_id: str | None = None,
    ) -> BenchmarkJob:
        if not parameter_sets:
            raise ValueError("experiment must contain at least one parameter set")
        if repetitions < 1:
            raise ValueError("experiment repetitions must be positive")
        resolved_job_id = job_id or uuid.uuid4().hex
        job_directory = self.jobs_root / resolved_job_id
        job_directory.mkdir(parents=True, exist_ok=False)
        first_parameters = parameter_sets[0]
        task_prefix = profile_for(first_parameters.command)["task_prefix"]
        return BenchmarkJob(
            job_id=resolved_job_id,
            parameters=first_parameters,
            task_label_prefix=f"{task_prefix}-ui-{resolved_job_id[:8]}-r001",
            config_path=(
                job_directory
                / "run-001"
                / f"{first_parameters.command}.yml"
            ),
            runner_log_path=job_directory / "run-001" / "runner.log",
            parameter_sets=parameter_sets,
            repetitions=repetitions,
            total_runs=len(parameter_sets) * repetitions,
        )

    def submit(
        self,
        parameter_sets: BenchmarkJobParameters | list[BenchmarkJobParameters],
        repetitions: int = 1,
    ) -> BenchmarkJob:
        with self._lock:
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active and active.status in ACTIVE_STATUSES:
                    raise BenchmarkJobConflictError(
                        f"benchmark job {active.job_id} is already {active.status}"
                    )
                self._active_job_id = None

            normalized_sets = (
                (parameter_sets,)
                if isinstance(parameter_sets, BenchmarkJobParameters)
                else tuple(parameter_sets)
            )
            job = self.prepare_experiment(normalized_sets, repetitions)
            self._jobs[job.job_id] = job
            self._active_job_id = job.job_id
            thread = threading.Thread(
                target=self._run_job,
                args=(job.job_id,),
                name=f"vectordbbench-{job.job_id[:8]}",
                daemon=True,
            )
            thread.start()
            return copy.copy(job)

    def _prepare_run_artifacts(
        self,
        job: BenchmarkJob,
        parameters: BenchmarkJobParameters,
        run_number: int,
    ) -> None:
        run_directory = self.jobs_root / job.job_id / f"run-{run_number:03d}"
        run_directory.mkdir(parents=True, exist_ok=True)
        task_prefix = profile_for(parameters.command)["task_prefix"]
        task_label_prefix = (
            f"{task_prefix}-ui-{job.job_id[:8]}-r{run_number:03d}"
        )
        config_path = run_directory / f"{parameters.command}.yml"
        config_path.write_text(
            yaml.safe_dump(
                self._build_job_config(parameters, task_label_prefix),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        with self._lock:
            job.parameters = parameters
            job.task_label_prefix = task_label_prefix
            job.config_path = config_path
            job.runner_log_path = run_directory / "runner.log"
            job.current_run_number = run_number
            job.phase = "starting"

    def _command(self, job: BenchmarkJob) -> list[str]:
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if not powershell:
            raise RuntimeError("PowerShell executable was not found")
        return [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.runner_path),
            "-Command",
            job.parameters.command,
            "-ConfigFile",
            str(job.config_path),
        ]

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                self._finish_cancelled(job)
                return
            job.status = "running"
            job.phase = "starting"
            job.started_at = datetime.now().astimezone()
            job.started_monotonic = monotonic()

        try:
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"
            creation_flags = 0
            if os.name == "nt":
                creation_flags = (
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    | subprocess.CREATE_NO_WINDOW
                )

            plan = [
                parameters
                for parameters in job.parameter_sets
                for _ in range(job.repetitions)
            ]
            for run_number, parameters in enumerate(plan, start=1):
                with self._lock:
                    if job.cancel_requested:
                        self._finish_cancelled(job)
                        return
                self._prepare_run_artifacts(
                    job,
                    parameters,
                    run_number,
                )
                with job.runner_log_path.open(
                    "w",
                    encoding="utf-8",
                    errors="replace",
                ) as runner_log:
                    process = subprocess.Popen(
                        self._command(job),
                        cwd=self.project_root,
                        env=environment,
                        stdout=runner_log,
                        stderr=subprocess.STDOUT,
                        creationflags=creation_flags,
                    )
                    with self._lock:
                        job.process_id = process.pid
                        self._processes[job_id] = process
                    exit_code = process.wait()
                with self._lock:
                    job.exit_code = exit_code
                    self._processes.pop(job_id, None)
                    job.process_id = None
                    if job.cancel_requested:
                        self._finish_cancelled(job)
                        return
                    if exit_code != 0:
                        raise RuntimeError(
                            f"run {run_number}/{job.total_runs} exited "
                            f"with code {exit_code}"
                        )
                    if not self._attach_result(job, job.task_label_prefix):
                        raise RuntimeError(
                            f"run {run_number}/{job.total_runs} finished, "
                            "but no matching SQLite result was found"
                        )
                    job.completed_runs = run_number
            with self._lock:
                job.status = "succeeded"
                job.phase = "completed"
                job.finished_at = datetime.now().astimezone()
        except Exception as error:
            with self._lock:
                job.status = "cancelled" if job.cancel_requested else "failed"
                job.phase = job.status
                job.error = None if job.cancel_requested else str(error)
                job.finished_at = datetime.now().astimezone()
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _attach_result(
        self,
        job: BenchmarkJob,
        task_label_prefix: str,
    ) -> bool:
        if not self.database_path.is_file():
            return False
        try:
            with sqlite3.connect(self.database_path) as connection:
                row = connection.execute(
                    """
                    SELECT run_id, case_index
                    FROM benchmark_runs
                    WHERE task_label LIKE ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (f"{task_label_prefix}-%",),
                ).fetchone()
            if row:
                job.result_run_id = str(row[0])
                job.result_case_index = int(row[1])
                job.result_run_ids.append(str(row[0]))
                return True
        except sqlite3.Error as error:
            job.error = f"benchmark succeeded but result lookup failed: {error}"
        return False

    def _finish_cancelled(self, job: BenchmarkJob) -> None:
        job.status = "cancelled"
        job.phase = "cancelled"
        job.finished_at = datetime.now().astimezone()
        if self._active_job_id == job.job_id:
            self._active_job_id = None

    def cancel(self, job_id: str) -> BenchmarkJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status in TERMINAL_STATUSES:
                return copy.copy(job)
            job.cancel_requested = True
            job.status = "cancelling"
            job.phase = "cancelling"
            process = self._processes.get(job_id)

        if process is not None and process.poll() is None:
            self._terminate_process_tree(process)
        return self.get(job_id)

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            process.terminate()

    def _phase_from_log(self, job: BenchmarkJob) -> str:
        if job.status != "running" or not job.runner_log_path.is_file():
            return job.phase
        text = job.runner_log_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        markers = [
            ("starting", "VectorDBBench command"),
            ("loading", "Start concurrent insert"),
            ("optimizing", "Milvus optimizing before search"),
            ("concurrent_search", "start concurrency search"),
            ("serial_search", "start serial search"),
            ("saving", "write results to disk"),
            ("collecting_metrics", "Formatted result JSON"),
            ("completed", "Saved benchmark metrics"),
        ]
        latest_phase = job.phase
        latest_position = -1
        lowered = text.lower()
        for phase, marker in markers:
            position = lowered.rfind(marker.lower())
            if position > latest_position:
                latest_position = position
                latest_phase = phase
        return latest_phase

    @staticmethod
    def _tail(path: Path, line_count: int = 30) -> list[str]:
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8", errors="replace") as file:
            return list(deque((line.rstrip() for line in file), maxlen=line_count))

    def get(self, job_id: str) -> BenchmarkJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.phase = self._phase_from_log(job)
            return copy.copy(job)

    def list(self) -> list[BenchmarkJob]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )
            return [self.get(job.job_id) for job in jobs]

    def log_tail(self, job_id: str) -> list[str]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            path = job.runner_log_path
        return self._tail(path)

    def elapsed_seconds(self, job: BenchmarkJob) -> float | None:
        if job.started_at is None:
            return None
        end = job.finished_at or datetime.now().astimezone()
        return max(0.0, (end - job.started_at).total_seconds())

    @property
    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active_job_id
