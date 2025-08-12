from typing import List
from vidur.events.base_event import BaseEvent
from vidur.types import EventType
from vidur.metrics import MetricsStore
from vidur.entities.request import Request

class ApiRequestCompletionEvent(BaseEvent):
    def __init__(self, time: float, request: Request, ttft_ms: float | None = None):
        super().__init__(time, EventType.API_REQUEST_COMPLETION)
        self._request = request
        self._ttft_ms = ttft_ms  # optional: for debugging/metrics tagging

    def handle_event(self, scheduler, metrics_store: MetricsStore) -> List[BaseEvent]:
        req = self._request

        # Mark as offloaded
        try:
            req.set_offloaded(True)
        except AttributeError:
            setattr(req, "_offloaded", True)

        # Ensure timestamps are consistent for metrics:
        # - scheduled_at: when we decided to offload (fallback to arrived_at)
        # - prefill_completed_at: treat as instant at scheduled_at for simplicity
        # - completed_at: now (this event's time)
        # if getattr(req, "scheduled_at", None) is None:
        #     try:
        #         req.scheduled_at = getattr(req, "arrived_at", self.time)
        #     except Exception:
        #         # Some Request implementations only expose properties; store privately if needed
        #         setattr(req, "_scheduled_at", getattr(req, "arrived_at", self.time))

        # # Mark prefill as completed so normalized metrics don’t divide by zero
        # try:
        #     req.prefill_completed_at = getattr(req, "scheduled_at", getattr(req, "_scheduled_at", self.time))
        #     req._is_prefill_complete = True
        # except Exception:
        #     setattr(req, "_is_prefill_complete", True)
        #     setattr(req, "prefill_completed_at", getattr(req, "scheduled_at", getattr(req, "_scheduled_at", self.time)))

        # # Final completion
        # try:
        #     req.completed = True
        #     req.completed_at = self.time
        # except Exception:
        #     setattr(req, "_completed", True)
        #     setattr(req, "_completed_at", self.time)

        # # If your Request tracks decode progression flags, make them consistent (best-effort)
        # for attr, val in [("has_started_decode", True), ("preempted", False)]:
        #     if not hasattr(req, attr):
        #         continue
        #     try:
        #         setattr(req, attr, val)
        #     except Exception:
        #         pass

        # Log request-level metrics (bypasses batch metrics)
        try:
            metrics_store._on_request_end(self.time, req)  # private but stable inside Vidur
        except Exception:
            # If write_metrics is disabled or structure differs, ignore silently
            pass

        # No follow-up events, external API “completes” here.
        return []