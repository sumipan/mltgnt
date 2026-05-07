"""
mltgnt.scheduler — ジョブディスパッチコア + YAML パーサ。

元コード: tools/secretary/scheduler.py のコア + YAML パーサ
SchedulerConfig 引数で受け取る。ペルソナ関連コールバックは __init__ 引数で注入。

設計: Issue #118 §3 (T4)
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import yaml
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from mltgnt.config import SchedulerConfig
    from mltgnt.interfaces.slack import SlackClientProtocol

__all__ = [
    "ScheduleJob",
    "SecretaryScheduler",
    "load_schedule_jobs",
    "atomic_write_text",
    "_hash_offset",
]

DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DEFAULT_TIMEZONE = "Asia/Tokyo"


def _parse_hhmm(s: str) -> tuple[int, int]:
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"HH:MM 形式ではありません: {s!r}")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"時刻が範囲外です: {s!r}")
    return h, m


def _to_minutes_since_midnight(h: int, m: int) -> int:
    return h * 60 + m


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _hash_offset(job_id: str, local_date: str, salt: str, span: int) -> int:
    if span <= 0:
        return 0
    payload = f"{job_id}|{local_date}|{salt}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:12], 16) % span


@dataclass
class ScheduleJob:
    id: str
    mode: str  # scheduled | fuzzy_window | interval | chained
    action: str
    notify: str
    timezone: str = _DEFAULT_TIMEZONE
    enabled: bool = True
    action_args: dict[str, Any] = field(default_factory=dict)
    every_day_at: Optional[str] = None
    interval_minutes: Optional[int] = None  # interval モード用（分単位）
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    fuzzy_method: str = "hash"  # hash | random
    on_window_missed: str = "notify"  # notify | silent | mark_done
    slack_channel: Optional[str] = None
    timeout_seconds: int = 600
    memory: bool = False
    persona: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    on_chain_failure: str = "abort_notify"  # abort_notify | silent
    every_week_on: Optional[str] = None  # "monday" | ... | "sunday" (None = daily)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default_timezone: str = _DEFAULT_TIMEZONE) -> "ScheduleJob":
        jid = str(raw["id"])
        mode = str(raw["mode"])
        action = str(raw["action"])
        notify = str(raw.get("notify", "silent"))
        tz = str(raw.get("timezone", default_timezone))
        enabled = bool(raw.get("enabled", True))
        args = raw.get("action_args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"job {jid}: action_args は辞書である必要があります")
        timeout = int(args.get("timeout_seconds", raw.get("timeout_seconds", 600)))

        every = raw.get("every_day_at")
        interval_min = raw.get("interval_minutes")
        ws = raw.get("window_start")
        we = raw.get("window_end")
        fuzzy_method = str(raw.get("fuzzy_method", "hash"))
        on_missed = str(raw.get("on_window_missed", "notify"))
        slack_ch = raw.get("slack_channel")
        slack_ch = str(slack_ch).strip() if slack_ch else None

        depends_on_raw = raw.get("depends_on") or []
        if isinstance(depends_on_raw, list):
            depends_on = [str(x) for x in depends_on_raw]
        else:
            depends_on = []
        on_chain_failure = str(raw.get("on_chain_failure", "abort_notify"))

        if mode == "scheduled":
            if not every:
                raise ValueError(f"job {jid}: scheduled には every_day_at が必要です")
            _parse_hhmm(str(every))
        elif mode == "interval":
            if not interval_min or int(interval_min) <= 0:
                raise ValueError(f"job {jid}: interval には interval_minutes > 0 が必要です")
        elif mode == "fuzzy_window":
            if not ws or not we:
                raise ValueError(f"job {jid}: fuzzy_window には window_start/end が必要です")
            sh, sm = _parse_hhmm(str(ws))
            eh, em = _parse_hhmm(str(we))
            s_min = _to_minutes_since_midnight(sh, sm)
            e_min = _to_minutes_since_midnight(eh, em)
            if s_min > e_min:
                raise ValueError(
                    f"job {jid}: 日付またぎのウィンドウは禁止です（window_start > window_end）"
                )
        elif mode == "chained":
            # chained: depends_on 全完了でトリガー。every_day_at はフォールバック用（任意）
            if every:
                _parse_hhmm(str(every))
        else:
            raise ValueError(f"job {jid}: 不明な mode: {mode}")

        if fuzzy_method not in ("hash", "random"):
            raise ValueError(f"job {jid}: fuzzy_method は hash か random です")
        if on_missed not in ("notify", "silent", "mark_done"):
            raise ValueError(f"job {jid}: on_window_missed が不正です")
        if notify not in ("silent", "slack_secretary", "slack_custom"):
            raise ValueError(f"job {jid}: notify が不正です")
        if notify == "slack_custom" and not slack_ch:
            raise ValueError(f"job {jid}: slack_custom には slack_channel が必要です")

        memory = bool(raw.get("memory", False))
        persona = raw.get("persona")
        persona = str(persona).strip() if persona else None

        every_week_on_raw = raw.get("every_week_on")
        every_week_on = str(every_week_on_raw).strip().lower() if every_week_on_raw else None
        if every_week_on and every_week_on not in DAY_NAMES:
            raise ValueError(
                f"job {jid}: every_week_on は {DAY_NAMES} のいずれかである必要があります: {every_week_on!r}"
            )

        return cls(
            id=jid,
            mode=mode,
            action=action,
            notify=notify,
            timezone=tz,
            enabled=enabled,
            action_args=args,
            every_day_at=str(every) if every else None,
            interval_minutes=int(interval_min) if interval_min else None,
            window_start=str(ws) if ws else None,
            window_end=str(we) if we else None,
            fuzzy_method=fuzzy_method,
            on_window_missed=on_missed,
            slack_channel=slack_ch,
            timeout_seconds=timeout,
            memory=memory,
            persona=persona,
            depends_on=depends_on,
            on_chain_failure=on_chain_failure,
            every_week_on=every_week_on,
        )

    def target_hhmm_scheduled(self) -> tuple[int, int]:
        assert self.every_day_at
        return _parse_hhmm(self.every_day_at)

    def window_minutes(self) -> tuple[int, int, int]:
        """戻り値: (start_min, end_min, span_minutes 閉区間)."""
        assert self.window_start and self.window_end
        sh, sm = _parse_hhmm(self.window_start)
        eh, em = _parse_hhmm(self.window_end)
        s_min = _to_minutes_since_midnight(sh, sm)
        e_min = _to_minutes_since_midnight(eh, em)
        span = e_min - s_min + 1
        return s_min, e_min, span


def load_schedule_jobs(
    yaml_path: Path,
    *,
    default_timezone: str = _DEFAULT_TIMEZONE,
) -> list[ScheduleJob]:
    if not yaml_path.is_file():
        return []
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    jobs_raw = data.get("jobs") or []
    if not isinstance(jobs_raw, list):
        raise ValueError("schedule.yaml: jobs はリストである必要があります")
    return [ScheduleJob.from_dict(j, default_timezone=default_timezone) for j in jobs_raw]


class SecretarySchedulePaths:
    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.done_dir = state_dir / "done"
        self.planned_dir = state_dir / "planned"
        self.missed_dir = state_dir / "missed"
        self.failed_dir = state_dir / "failed"

    def done_path(self, job_id: str, d: date) -> Path:
        return self.done_dir / f"{job_id}_{d.isoformat()}.done"

    def planned_path(self, job_id: str, d: date) -> Path:
        return self.planned_dir / f"{job_id}_{d.isoformat()}.json"

    def missed_path(self, job_id: str, d: date) -> Path:
        return self.missed_dir / f"{job_id}_{d.isoformat()}.flag"

    def failed_path(self, job_id: str, d: date) -> Path:
        return self.failed_dir / f"{job_id}_{d.isoformat()}.failed"


class SecretaryScheduler:
    """
    1 秒周期で tick する想定。main からデーモンスレッドで起動する。

    ペルソナ関連コールバックは __init__ 引数で注入（OSS 分離）。
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
        execute_action_fn: Optional[Callable[["SecretaryScheduler", ScheduleJob], tuple[bool, str]]] = None,
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
        self.paths = SecretarySchedulePaths(self._state_dir)
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
        self._execute_action_fn = execute_action_fn

        from mltgnt.skill import discover
        self._skill_registry = discover(paths=[self.repo_root / "skills"])

    def _load_jobs(self) -> list[ScheduleJob]:
        if self._jobs_override is not None:
            return [j for j in self._jobs_override if j.enabled]
        try:
            return [j for j in load_schedule_jobs(self._yaml_path, default_timezone=self._default_tz) if j.enabled]
        except Exception as e:
            print(f"[secretary-schedule] YAML 読込エラー: {e}", file=sys.stderr)
            return []

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
            jobs = self._load_jobs()
            cycled = self._detect_cycles(jobs)
            if cycled:
                print(f"[secretary-schedule] 循環依存を検知。無効化: {cycled}", file=sys.stderr)
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
            print(
                f"[secretary-schedule] メモリ記録失敗 {job.id}: {e}",
                file=sys.stderr,
            )


    def _post(self, job: ScheduleJob, text: str) -> None:
        if not self._should_notify_slack(job):
            return
        if self.slack is None:
            print(
                f"[secretary-schedule] Slack 未初期化のため通知スキップ: {text}",
                file=sys.stderr,
            )
            return
        if job.persona and self._persona_post_kwargs_resolver is not None:
            try:
                post_kwargs, resolved_text = self._persona_post_kwargs_resolver(job.persona, self.repo_root)
                if not text and resolved_text:
                    text = resolved_text
            except Exception as e:
                print(f"[secretary-schedule] ペルソナ読込失敗 {job.persona}: {e}", file=sys.stderr)
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
            print(
                f"[secretary-schedule] missed (silent): {job.id} {d.isoformat()}",
                file=sys.stderr,
            )
        if job.on_window_missed == "mark_done":
            self._mark_done(job, d)

    def build_command(self, job: ScheduleJob) -> list[str]:
        argv_extra: list[str] = []
        aa = job.action_args
        if isinstance(aa.get("argv"), list):
            argv_extra = [str(x) for x in aa["argv"]]

        if job.action == "noop":
            return []

        if job.action == "subprocess_diary_draft":
            cmd = [sys.executable, str(self.repo_root / "scripts" / "diary-draft.py")]
            if job.notify == "silent":
                cmd.append("--no-slack")
            cmd.extend(argv_extra)
            return cmd

        if job.action == "subprocess_from_slack":
            cmd = [
                sys.executable,
                str(self.repo_root / "scripts" / "diary-draft.py"),
                "--from-slack",
            ]
            if job.notify == "silent":
                cmd.append("--no-slack")
            cmd.extend(argv_extra)
            return cmd

        if job.action == "skill":
            return []

        if job.action == "append_exec_order":
            return []

        if job.action == "subprocess_script":
            script = aa.get("script")
            if not script:
                raise ValueError(f"job {job.id}: subprocess_script には action_args.script が必須です")
            cmd = [sys.executable, str(self.repo_root / "scripts" / script)]
            cmd.extend(argv_extra)
            return cmd

        raise ValueError(f"未対応の action: {job.action}")

    def execute_action(self, job: ScheduleJob) -> tuple[bool, str]:
        if self._execute_action_fn is not None:
            return self._execute_action_fn(self, job)

        if job.action == "noop":
            print(f"[secretary-schedule] noop: {job.id}", file=sys.stderr)
            return True, ""

        if job.action == "skill":
            aa = job.action_args
            skill_name = aa.get("skill")
            if not skill_name:
                return False, f"job {job.id}: action_args.skill が未指定です"
            persona_name = aa.get("persona")
            if not persona_name:
                return False, f"job {job.id}: action_args.persona が未指定です"

            # ペルソナは agents/ 配下の正規 persona ファイルを mltgnt.persona 経由で読む。
            # 旧実装は chat/memory/{name}.md を直接読んでいたが、そこはチャットログで
            # YAML frontmatter を持たないため engine/model が解決できずバグっていた。
            from mltgnt.persona import load_persona
            try:
                persona = load_persona(persona_name, persona_dir=self.persona_dir)
            except FileNotFoundError as e:
                return False, f"ペルソナファイルが見つかりません: {e}"
            except Exception as e:
                return False, f"ペルソナ読込失敗 {persona_name}: {e}"

            engine = aa.get("engine") or (persona.fm.engine or None)
            model = aa.get("model") or (persona.fm.model or None)

            meta = self._skill_registry.get(skill_name)
            if meta is None:
                return False, f"スキルが見つかりません: {skill_name}"

            from mltgnt.skill import load
            skill_file = load(meta)

            argv_list = aa.get("argv", [])
            argv_str = " ".join(str(x) for x in argv_list) if argv_list else ""

            from mltgnt.skill import runner as skill_runner
            from mltgnt.chat.models import ChatInput, Message

            chat_input = ChatInput(
                source="scheduler",
                session_key=job.id,
                messages=[Message(role="user", content=argv_str or "")],
                persona_name=persona.name,
                model=model,
            )
            chat_input = skill_runner.run(skill_file, persona, argv_str, chat_input)

            # runner.run() が system メッセージにプロンプトを格納する
            prompt = next(m["content"] for m in chat_input.messages if m["role"] == "system")
            resolved_model = chat_input.model  # skill.meta.model が優先される

            from mltgnt.scheduler.ghdag_bridge import enqueue_and_wait

            fired_at = datetime.now(ZoneInfo(self._default_tz))
            ok, msg = enqueue_and_wait(
                prompt=prompt,
                engine=engine,
                model=resolved_model,
                timeout=job.timeout_seconds or 120,
                idempotency_key=f"scheduler:{job.id}:{fired_at.isoformat()}",
                jobs_dir=self.repo_root / "jobs",
                exec_done_dir=self.repo_root / "jobs" / "done",
            )
            return ok, msg

        if job.action == "append_exec_order":
            dry_run = bool(job.action_args.get("dry_run", False))
            issue_number = int(job.action_args.get("issue_number", 0))
            try:
                _stash_dev = str(self.repo_root / "tools" / "stash-developer")
                if _stash_dev not in sys.path:
                    sys.path.insert(0, _stash_dev)
                from stash_developer.secretary_api import enqueue_issue  # noqa: PLC0415
                enqueue_issue(issue_number=issue_number, dry_run=dry_run)
                print(
                    f"[secretary-schedule] append_exec_order 完了: {job.id}",
                    file=sys.stderr,
                )
                return True, ""
            except Exception as e:
                return False, str(e)

        cmd = self.build_command(job)
        if not cmd:
            return False, "empty command"
        try:
            r = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=job.timeout_seconds,
            )
            err = (r.stderr or "").strip() or (r.stdout or "").strip()
            if r.returncode != 0:
                tail = err[-800:] if err else f"exit {r.returncode}"
                return False, tail
            return True, (r.stdout or "").strip()
        except subprocess.TimeoutExpired:
            return False, f"timeout ({job.timeout_seconds}s)"
        except OSError as e:
            return False, str(e)

    def _spawn_job(self, job: ScheduleJob, d: date, on_finish: Optional[Callable[[], None]] = None) -> None:
        def runner() -> None:
            fired_at = datetime.now(ZoneInfo(self._default_tz))
            try:
                ok, msg = self.execute_action(job)
                if ok:
                    if job.mode != "interval":
                        self._mark_done(job, d)
                    print(f"[secretary-schedule] 成功: {job.id}", file=sys.stderr)
                    if msg:
                        self._post(job, msg)
                    self._record_to_memory(job, msg, True, fired_at)
                else:
                    if job.mode != "interval":
                        self._mark_failed(job, d, reason=msg[:400])
                    print(f"[secretary-schedule] 失敗: {job.id}: {msg}", file=sys.stderr)
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
        print(
            f"[secretary-schedule] スレッド開始 jobs={len(self._jobs)} "
            f"yaml={self.yaml_path}",
            file=sys.stderr,
        )
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:
                print(f"[secretary-schedule] tick 例外: {e}", file=sys.stderr)
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
