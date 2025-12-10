import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import cast

from a2a.server.context import ServerCallContext
from a2a.server.events import Event
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.types import (DeleteTaskPushNotificationConfigParams,
                       GetTaskPushNotificationConfigParams,
                       ListTaskPushNotificationConfigParams, Message,
                       MessageSendParams, Task, TaskIdParams,
                       TaskPushNotificationConfig, TaskQueryParams, TaskState)
from a2a.utils.telemetry import SpanKind, trace_class
from transports.interactions_api import InteractionsApiTransport

logger = logging.getLogger(__name__)

TERMINAL_TASK_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


@trace_class(kind=SpanKind.SERVER)
class InteractionsAPIProxyRequestHandler(RequestHandler):
    """Request handler for all incoming requests to Interactions API Proxy.

    This handler provides default implementations for all A2A JSON-RPC methods,
    coordinating between the `AgentExecutor`, `TaskStore`, `QueueManager`,
    and optional `PushNotifier`.
    """

    _running_agents: dict[str, asyncio.Task]
    _background_tasks: set[asyncio.Task]

    def __init__(
            self,
            interactions_api_transport: InteractionsApiTransport,
    ) -> None:
        """Initializes the InteractionsAPIProxyRequestHandler.

        Args:
            interactions_api_transport: Interactions API Transport object.
        """
        self.interactions_api_transport = interactions_api_transport

    async def on_get_task(
        self,
        params: TaskQueryParams,
        context: ServerCallContext | None = None,
    ) -> Task | None:
        return self.interactions_api_transport.get_task(params)


    async def on_cancel_task(
        self, params: TaskIdParams, context: ServerCallContext | None = None
    ) -> Task | None:
        return self.interactions_api_transport.cancel_task(params)

    async def on_message_send(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> Message | Task:
        return self.interactions_api_transport.send_message(params)

    async def on_message_send_stream(
        self,
        params: MessageSendParams,
        context: ServerCallContext | None = None,
    ) -> AsyncGenerator[Event]:
       return self.interactions_api_transport.send_message_streaming(params)

    async def on_set_task_push_notification_config(
        self,
        params: TaskPushNotificationConfig,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        return self.interactions_api_transport.set_task_callback(params)

    async def on_get_task_push_notification_config(
        self,
        params: TaskIdParams | GetTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> TaskPushNotificationConfig:
        return self.interactions_api_transport.get_task_callback(params)

    async def on_resubscribe_to_task(
        self,
        params: TaskIdParams,
        context: ServerCallContext | None = None,
    ) -> AsyncGenerator[Event]:
       return self.interactions_api_transport.resubscribe(params)

    async def on_list_task_push_notification_config(
        self,
        params: ListTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> list[TaskPushNotificationConfig]:
        return self.interactions_api_transport.list_task_callback(params)

    async def on_delete_task_push_notification_config(
        self,
        params: DeleteTaskPushNotificationConfigParams,
        context: ServerCallContext | None = None,
    ) -> None:
       return self.interactions_api_transport.delete_task_callback(params)
