from typing import List

from vidur.entities import Request
from vidur.events.base_event import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class RequestArrivalEvent(BaseEvent):
    def __init__(self, time: float, request: Request) -> None:
        super().__init__(time, EventType.REQUEST_ARRIVAL)

        self._request = request

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.global_schedule_event import GlobalScheduleEvent

        logger.info(f"Request: {self._request.id} arrived at {self.time}, with prefill token {self._request.num_prefill_tokens} and decode token {self._request.num_decode_tokens}")
        scheduler.add_request(self._request)
        metrics_store.on_request_arrival(self.time, self._request)
        res = [GlobalScheduleEvent(self.time)]
        # print(f"L0 [GlobalScheduleEvent(self.time)] generated: {res} at {self.time}")
        return res

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type,
            "request": self._request.id,
        }
