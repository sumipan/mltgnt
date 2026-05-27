from __future__ import annotations

import json
import logging
import random
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from zoneinfo import ZoneInfo

from mltgnt.exceptions import ConfigError
from mltgnt.scheduler.actions.skill import run_skill_action
from mltgnt.scheduler.loader import load_schedule_jobs
from mltgnt.scheduler.models import (
    DAY_NAMES,
    ActionFn,
    ScheduleJob,
    _DEFAULT_TIMEZONE,
    _parse_hhmm,
    _to_minutes_since_midnight,
)
from mltgnt.scheduler.state import SchedulePaths, _hash_offset, atomic_write_text

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mltgnt.config import SchedulerConfig
    from mltgnt.interfaces.slack import SlackClientProtocol

class PersonaScheduler:
    """
    1 秒周期で tick する想定。main からデーモンスレッドで起動する。

    ペルソナ関連コールバックは __init__ 引数で注入（OSS 分離）。
    host 固有 action は actions= / register_action() で登録する。
    """

    def __init__(
        self,
        slack: Optional["SlackClientProtocol"],
        *,
        config: Optional["SchedulerConfig"] = None,
        state_dir: Optional[Path] = None,
        yaml_path: Optional[Path] = None,
        salt: str = "",
        jobs: Optional[list[ScheduleJob]] = None,
        notify_channel_resolver: Optional[Callable[[ScheduleJob], str]] = None,
        default_slack_post_kwargs: Optional[Callable[[], dict]] = None,
        persona_post_kwargs_resolver: Optional[Callable[[str, Path], tuple[dict, str]]] = None,
        repo_root: Optional[Path] = None,
        persona_dir: Optional[Path] = None,
        append_memory_fn: Optional[Callable[..., bool]] = None,
        actions: Optional[dict[str, "ActionFn"]] = None,
    ) -> None:
        # config が指定された場合、config から設定を取得
        if config is not None:
            self._state_dir = state_dir or config.state_dir
            self._yaml_path = yaml_path or config.schedule_yaml
            self._salt = salt or config.salt
            self._default_tz = config.timezone
        else:
            self._state_dir = state_dir or Path(".secretary_schedule_state")
            self._yaml_path = yaml_path or Path("schedule.yaml")
            self._salt = salt
            self._default_tz = _DEFAULT_TIMEZONE

        self.slack = slack
        self.state_dir = self._state_dir
        self.yaml_path = self._yaml_path
        self.salt = self._salt
        self.repo_root = repo_root or Path(".")
        self.persona_dir = persona_dir or (self.repo_root / "agents")
        self.paths = SchedulePaths(self._state_dir)
        self._jobs_override = jobs
        self._reload_counter = 0
        self._jobs: list[ScheduleJob] = []
        self._jobs_lock = threading.Lock()
        self._calendar_date: Optional[date] = None
        self._scheduled_fired_slot: dict[str, tuple[date, int, int]] = {}
        self._fuzzy_last_dispatch_slot: dict[str, tuple[date, int, int]] = {}
        self._interval_last_fired: dict[str, datetime] = {}
        self._running: set[str] = set()
        self._run_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Injected callbacks for OSS separation
        self._notify_channel_resolver = notify_channel_resolver
        self._default_slack_post_kwargs = default_slack_post_kwargs
        self._persona_post_kwargs_resolver = persona_post_kwargs_resolver
        self._append_memory_fn = append_memory_fn
        self._actions: dict[str, ActionFn] = dict(actions or {})

        from mltgnt.skill import discover
        self._skill_registry = discover(paths=[self.repo_root / "skills"])

    def register_action(self, name: str, fn: "ActionFn") -> None:
        self._actions[name] = fn

    def _load_jobs(self) -> list[ScheduleJob]:
        if self._jobs_override is not None:
            return [j for j in self._jobs_override if j.enabled]
        try:
            return [j for j in load_schedule_jobs(self._yaml_path, default_timezone=self._default_tz) if j.enabled]
        except Exception as e:
            raise ConfigError(f"YAML 読込エラー: {e}") from e

    def _detect_cycles(self, jobs: list[ScheduleJob]) -> list[str]:
        """循環依存があるジョブIDのリストを返す。"""
        dep_map = {j.id: j.depends_on for j in jobs}

        def has_cycle(start: str, path: set[str]) -> bool:
            if start in path:
                return True
            path.add(start)
            for dep in dep_map.get(start, []):
                if dep in dep_map and has_cycle(dep, path):
                    return True
            path.discard(start)
            return False

        cycled = []
        for jid in dep_map:
            if has_cycle(jid, set()):
                cycled.append(jid)
        return cycled

    def reload_jobs(self) -> None:
        with self._jobs_lock:
            try:
                jobs = self._load_jobs()
            except ConfigError as e:
                _log.error("%s", e)
                return
            cycled = self._detect_cycles(jobs)
            if cycled:
                _log.warning("循環依存を検知。無効化: %s", cycled)
                jobs = [j for j in jobs if j.id not in cycled]
            self._jobs = jobs

    def notify_channel(self, job: ScheduleJob) -> str:
        if job.notify == "slack_custom" and job.slack_channel:
            return job.slack_channel
        if self._notify_channel_resolver is not None:
            return self._notify_channel_resolver(job)
        return ""

    def _should_notify_slack(self, job: ScheduleJob) -> bool:
        return job.notify in ("slack_secretary", "slack_custom")

    def _record_to_memory(
        self,
        job: ScheduleJob,
        post_text: str,
        success: bool,
        fired_at: datetime,
    ) -> None:
        """スケジュールタスクの実行結果をメモリに記録する。

        条件: job.memory is True AND job.notify が slack_*。
        例外は warning ログで吸収し、Slack 投稿済みの状態を汚染しない。
        """
        if not job.memory:
            return
        if not self._should_notify_slack(job):
            return
        if self._append_memory_fn is None:
            return
        persona_stem = job.persona or ""
        if not persona_stem:
            return
        ts = fired_at.strftime("%Y-%m-%d %H:%M")
        dedupe_base = f"scheduled:{job.id}:{fired_at.isoformat(timespec='seconds')}"
        content_assistant = ("[FAILED] " if not success else "") + (
            post_text or ("スケジュールタスクが完了しました" if success else "失敗しました")
        )
        try:
            self._append_memory_fn(
                persona_stem,
                "user",
                f"[スケジュールタスク: {job.id}]",
                ts,
                source_tag="[scheduled]",
                dedupe_key=f"{dedupe_base}:user",
            )
            self._append_memory_fn(
                persona_stem,
                "assistant",
                content_assistant,
                ts,
                source_tag="[scheduled]",
                dedupe_key=f"{dedupe_base}:assistant",
            )
        except Exception as e:
            _log.warning("メモリ記録失敗 %s: %s", job.id, e)


    def _post(self, job: ScheduleJob, text: str) -> None:
        if not self._should_notify_slack(job):
            return
        if self.slack is None:
            _log.warning("Slack 未初期化のため通知スキップ: %s", text)
            return
        if job.persona and self._persona_post_kwargs_resolver is not None:
            try:
                post_kwargs, resolved_text = self._persona_post_kwargs_resolver(job.persona, self.repo_root)
                if not text and resolved_text:
                    text = resolved_text
            except Exception as e:
                _log.warning("ペルソナ読込失敗 %s: %s", job.persona, e)
                post_kwargs = self._default_slack_post_kwargs() if self._default_slack_post_kwargs else {}
        else:
            post_kwargs = self._default_slack_post_kwargs() if self._default_slack_post_kwargs else {}
        self.slack.post_message(
            text,
            channel=self.notify_channel(job),
            **post_kwargs,
        )

    def _mark_done(self, job: ScheduleJob, d: date) -> None:
        p = self.paths.done_path(job.id, d)
        atomic_write_text(p, f"ok\t{datetime.now().isoformat(timespec='seconds')}\n")

    def _is_done(self, job: ScheduleJob, d: date) -> bool:
        return self.paths.done_path(job.id, d).is_file()

    def _mark_failed(self, job: ScheduleJob, d: date, reason: str = "") -> None:
        p = self.paths.failed_path(job.id, d)
        line = f"failed\t{datetime.now().isoformat(timespec='seconds')}"
        if reason:
            line += f"\t{reason}"
        atomic_write_text(p, line + "\n")

    def _is_failed(self, job: ScheduleJob, d: date) -> bool:
        return self.paths.failed_path(job.id, d).is_file()

    def _read_failed_reason(self, job_id: str, d: date) -> str:
        """failed マーカーからエラー理由を読み取る。なければ空文字。"""
        p = self.paths.failed_path(job_id, d)
        try:
            parts = p.read_text(encoding="utf-8").strip().split("\t", 2)
            return parts[2] if len(parts) >= 3 else ""
        except OSError:
            return ""

    def _find_job(self, job_id: str) -> Optional[ScheduleJob]:
        with self._jobs_lock:
            for j in self._jobs:
                if j.id == job_id:
                    return j
        return None

    def _check_depends(self, job: ScheduleJob, d: date) -> str:
        """Returns: 'ok' | 'pending' | 'failed'"""
        for dep_id in job.depends_on:
            dep_done_path = self.paths.done_path(dep_id, d)
            dep_failed_path = self.paths.failed_path(dep_id, d)
            if dep_failed_path.is_file():
                return "failed"
            if not dep_done_path.is_file():
                return "pending"
        return "ok"

    def _chain_failure_text(self, job: ScheduleJob, d: date) -> str:
        """chain failure 通知用テキスト。依存先の失敗理由があれば含める。"""
        reasons: list[str] = []
        for dep_id in job.depends_on:
            if self.paths.failed_path(dep_id, d).is_file():
                r = self._read_failed_reason(dep_id, d)
                if r:
                    reasons.append(f"{dep_id}: {r[:200]}")
        text = f"[secretary-schedule] 依存ジョブ失敗のためスキップ: `{job.id}` (chain failure)"
        if reasons:
            detail = "\n".join(reasons)
            text += f"\n```\n{detail}\n```"
        return text

    def _read_planned_minute(self, job: ScheduleJob, d: date) -> Optional[int]:
        path = self.paths.planned_path(job.id, d)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data["run_minute"])
        except Exception:
            return None

    def _write_planned_minute(self, job: ScheduleJob, d: date, run_minute: int) -> None:
        path = self.paths.planned_path(job.id, d)
        atomic_write_text(path, json.dumps({"run_minute": run_minute}, ensure_ascii=False))

    def _ensure_planned_fuzzy(self, job: ScheduleJob, d: date, z: ZoneInfo) -> int:
        existing = self._read_planned_minute(job, d)
        if existing is not None:
            return existing
        start_m, end_m, span = job.window_minutes()
        salt = self.salt or "default_salt"
        if job.fuzzy_method == "hash":
            off = _hash_offset(job.id, d.isoformat(), salt, span)
        else:
            off = random.randrange(span)
        run_minute = start_m + off
        self._write_planned_minute(job, d, run_minute)
        return run_minute

    def _maybe_handle_missed_fuzzy(
        self,
        job: ScheduleJob,
        d: date,
        now_min: int,
        end_m: int,
    ) -> None:
        if now_min <= end_m:
            return
        if self._is_done(job, d):
            return
        if self._is_failed(job, d):
            return
        missed_p = self.paths.missed_path(job.id, d)
        if missed_p.is_file():
            return
        atomic_write_text(missed_p, "missed\n")
        if job.on_window_missed == "notify":
            self._post(
                job,
                f"[secretary-schedule] ウィンドウ内に完了しませんでした（missed）: `{job.id}` {d.isoformat()}",
            )
        elif job.on_window_missed == "silent":
            _log.info("missed (silent): %s %s", job.id, d.isoformat())
        if job.on_window_missed == "mark_done":
            self._mark_done(job, d)

    def build_command(self, job: ScheduleJob) -> list[str]:
        if job.action in ("noop", "skill"):
            return []
        raise ValueError(f"未対応の action: {job.action}")

    def execute_action(self, job: ScheduleJob) -> tuple[bool, str]:
        if job.action == "noop":
            _log.debug("noop: %s", job.id)
            return True, ""

        if job.action == "skill":
            return run_skill_action(
                job,
                persona_dir=self.persona_dir,
                skill_registry=self._skill_registry,
                default_tz=self._default_tz,
                repo_root=self.repo_root,
            )

        if job.action in self._actions:
            return self._actions[job.action](job)

        raise ValueError(f"未対応の action: {job.action}")

    def _spawn_job(self, job: ScheduleJob, d: date, on_finish: Optional[Callable[[], None]] = None) -> None:
        def runner() -> None:
            fired_at = datetime.now(ZoneInfo(self._default_tz))
            try:
                ok, msg = self.execute_action(job)
                if ok:
                    if job.mode != "interval":
                        self._mark_done(job, d)
                    _log.info("成功: %s", job.id)
                    if msg:
                        self._post(job, msg)
                    self._record_to_memory(job, msg, True, fired_at)
                else:
                    if job.mode != "interval":
                        self._mark_failed(job, d, reason=msg[:400])
                    _log.error("失敗: %s: %s", job.id, msg)
                    snippet = msg.strip()[-400:] if msg.strip() else "(詳細なし)"
                    if len(msg.strip()) > 400:
                        snippet = "…" + snippet
                    fail_text = (
                        f"[secretary-schedule] ジョブ失敗 `{job.id}`\n"
                        f"```\n{snippet}\n```"
                    )
                    self._post(job, fail_text)
                    self._record_to_memory(job, fail_text, False, fired_at)
            finally:
                with self._run_lock:
                    self._running.discard(job.id)
                if on_finish:
                    on_finish()

        with self._run_lock:
            if job.id in self._running:
                return
            self._running.add(job.id)
        threading.Thread(target=runner, name=f"sec-sched-{job.id}", daemon=True).start()

    def tick(self, now: Optional[datetime] = None) -> None:
        if now is None:
            now = datetime.now(ZoneInfo(self._default_tz))
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ZoneInfo(self._default_tz))

        self._reload_counter += 1
        if self._reload_counter >= 60:
            self._reload_counter = 0
            self.reload_jobs()

        with self._jobs_lock:
            jobs = list(self._jobs)

        tick_date = now.astimezone(ZoneInfo(self._default_tz)).date()
        if self._calendar_date != tick_date:
            self._calendar_date = tick_date
            self._scheduled_fired_slot.clear()
            self._fuzzy_last_dispatch_slot.clear()

        for job in jobs:
            z = ZoneInfo(job.timezone)
            local = now.astimezone(z)
            d = local.date()
            now_min = _to_minutes_since_midnight(local.hour, local.minute)

            if job.mode != "interval":
                if self._is_done(job, d):
                    continue
                if self._is_failed(job, d):
                    continue

            if job.mode == "scheduled":
                th, tm = job.target_hhmm_scheduled()
                if local.hour != th or local.minute != tm:
                    continue
                if job.every_week_on:
                    expected_weekday = DAY_NAMES.index(job.every_week_on)
                    if local.weekday() != expected_weekday:
                        continue
                slot = (d, th, tm)
                if self._scheduled_fired_slot.get(job.id) == slot:
                    continue
                dep_status = self._check_depends(job, d)
                if dep_status == "failed":
                    if not self._is_failed(job, d):
                        self._mark_failed(job, d)
                        if job.on_chain_failure == "abort_notify":
                            self._post(job, self._chain_failure_text(job, d))
                    continue
                if dep_status == "pending":
                    continue
                with self._run_lock:
                    if job.id in self._running:
                        continue
                self._scheduled_fired_slot[job.id] = slot
                self._spawn_job(job, d)
                continue

            if job.mode == "interval":
                assert job.interval_minutes
                last = self._interval_last_fired.get(job.id)
                if last is not None:
                    elapsed = (now - last).total_seconds() / 60
                    if elapsed < job.interval_minutes:
                        continue
                with self._run_lock:
                    if job.id in self._running:
                        continue
                self._interval_last_fired[job.id] = now
                self._spawn_job(job, d)
                continue

            if job.mode == "fuzzy_window":
                start_m, end_m, _span = job.window_minutes()
                self._maybe_handle_missed_fuzzy(job, d, now_min, end_m)
                if self._is_done(job, d):
                    continue
                if now_min < start_m or now_min > end_m:
                    continue
                run_minute = self._ensure_planned_fuzzy(job, d, z)
                if now_min < run_minute:
                    continue
                dep_status = self._check_depends(job, d)
                if dep_status == "failed":
                    if not self._is_failed(job, d):
                        self._mark_failed(job, d)
                        if job.on_chain_failure == "abort_notify":
                            self._post(job, self._chain_failure_text(job, d))
                    continue
                if dep_status == "pending":
                    continue
                with self._run_lock:
                    if job.id in self._running:
                        continue
                slot = (d, local.hour, local.minute)
                if self._fuzzy_last_dispatch_slot.get(job.id) == slot:
                    continue
                self._fuzzy_last_dispatch_slot[job.id] = slot
                self._spawn_job(job, d)
                continue

            if job.mode == "chained":
                dep_status = self._check_depends(job, d)
                if dep_status == "failed":
                    if not self._is_failed(job, d):
                        self._mark_failed(job, d)
                        if job.on_chain_failure == "abort_notify":
                            self._post(job, self._chain_failure_text(job, d))
                    continue
                if dep_status == "ok":
                    with self._run_lock:
                        if job.id in self._running:
                            continue
                    slot = (d, local.hour, local.minute)
                    if self._fuzzy_last_dispatch_slot.get(job.id) == slot:
                        continue
                    self._fuzzy_last_dispatch_slot[job.id] = slot
                    self._spawn_job(job, d)
                    continue
                # pending: fallback to every_day_at if defined
                if job.every_day_at:
                    th, tm = _parse_hhmm(job.every_day_at)
                    if local.hour == th and local.minute == tm:
                        slot = (d, th, tm)
                        if self._scheduled_fired_slot.get(job.id) != slot:
                            with self._run_lock:
                                if job.id in self._running:
                                    continue
                            self._scheduled_fired_slot[job.id] = slot
                            self._spawn_job(job, d)
                continue

    def loop(self) -> None:
        self.reload_jobs()
        _log.info(
            "スレッド開始 jobs=%d yaml=%s",
            len(self._jobs),
            self.yaml_path,
        )
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                _log.exception("tick 例外")
            time.sleep(1.0)

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _run() -> None:
            self.loop()

        self._thread = threading.Thread(target=_run, name="secretary-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
