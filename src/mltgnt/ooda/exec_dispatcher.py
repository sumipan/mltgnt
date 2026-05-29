"""mltgnt.ooda.exec_dispatcher — exec.jsonl 追記と Slack 通知の ActDispatcher 実装。"""
from __future__ import annotations

import json
import logging
import uuid as _uuid_module
from pathlib import Path
from typing import Any

from mltgnt.interfaces.ooda import ActResult
from mltgnt.interfaces.slack import SlackClientProtocol

_logger = logging.getLogger(__name__)


class ExecAppenderDispatcher:
    """recover_task / escalate_to_slack / skip を実行する ActDispatcher。"""

    def __init__(
        self,
        *,
        exec_jsonl_path: Path,
        slack_client: SlackClientProtocol | None = None,
        default_channel: str = "",
        logger: logging.Logger | None = None,
    ) -> None:
        self._exec_jsonl_path = exec_jsonl_path
        self._slack_client = slack_client
        self._default_channel = default_channel
        self._logger = logger or _logger

    def dispatch(self, action: str, args: dict[str, Any]) -> ActResult:
        if action == "skip":
            return ActResult(action="skip", success=True, detail="skipped")

        if action == "recover_task":
            return self._recover_task(args)

        if action == "escalate_to_slack":
            return self._escalate_to_slack(args)

        return ActResult(action=action, success=False, detail=f"unknown action: {action}")

    def _recover_task(self, args: dict[str, Any]) -> ActResult:
        record_uuid = args.get("uuid") or str(_uuid_module.uuid4())
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return ActResult(action="recover_task", success=False, detail="missing command")

        result_path = args.get("result_path", "")
        if not isinstance(result_path, str):
            result_path = ""

        depends = args.get("depends", [])
        if not isinstance(depends, list):
            depends = []

        record: dict[str, Any] = {
            "uuid": str(record_uuid),
            "command": command,
            "depends": depends,
            "result_path": result_path,
            "retry": int(args.get("retry", 0)),
            "annotations": args.get("annotations") if isinstance(args.get("annotations"), dict) else {},
        }
        idempotency_key = args.get("idempotency_key")
        if isinstance(idempotency_key, str) and idempotency_key:
            record["idempotency_key"] = idempotency_key

        self._exec_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with self._exec_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

        return ActResult(action="recover_task", success=True, detail=f"appended exec record {record_uuid}")

    def _escalate_to_slack(self, args: dict[str, Any]) -> ActResult:
        text = args.get("text") or args.get("message")
        if not isinstance(text, str) or not text.strip():
            return ActResult(action="escalate_to_slack", success=False, detail="missing text")

        channel = args.get("channel") or self._default_channel
        if not isinstance(channel, str) or not channel:
            return ActResult(action="escalate_to_slack", success=False, detail="missing channel")

        if self._slack_client is None:
            return ActResult(action="escalate_to_slack", success=False, detail="slack client not configured")

        ok = self._slack_client.post_message(text=text, channel=channel)
        if ok:
            return ActResult(action="escalate_to_slack", success=True, detail="posted to slack")
        return ActResult(action="escalate_to_slack", success=False, detail="slack post failed")
