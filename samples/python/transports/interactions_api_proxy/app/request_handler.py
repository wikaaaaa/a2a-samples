import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import cast

from a2a.server.agent_execution import (AgentExecutor, RequestContext,
                                        RequestContextBuilder,
                                        SimpleRequestContextBuilder)
from a2a.server.context import ServerCallContext
from a2a.server.events import (Event, EventConsumer, EventQueue,
                               InMemoryQueueManager, QueueManager)
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.tasks import (PushNotificationConfigStore,
                              PushNotificationSender, ResultAggregator,
                              TaskManager, TaskStore)
from a2a.types import (DeleteTaskPushNotificationConfigParams,
                       GetTaskPushNotificationConfigParams, InternalError,
                       InvalidParamsError,
                       ListTaskPushNotificationConfigParams, Message,
                       MessageSendParams, Task, TaskIdParams,
                       TaskNotCancelableError, TaskNotFoundError,
                       TaskPushNotificationConfig, TaskQueryParams, TaskState,
                       UnsupportedOperationError)
from a2a.utils.errors import ServerError
from a2a.utils.task import apply_history_length
from a2a.utils.telemetry import SpanKind, trace_class
from interactions_api import InteractionsApiTransport

logger = logging.getLogger(__name__)

TERMINAL_TASK_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


@trace_class(kind=SpanKind.SERVER)
class InteractionsAPIProxyRequestHandler(RequestHandler):
    """Default request handler for all incoming requests.

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
        """Initializes the DefaultRequestHandler.

        Args:
            agent_executor: The `AgentExecutor` instance to run agent logic.
            task_store: The `TaskStore` instance to manage task persistence.
            queue_manager: The `QueueManager` instance to manage event queues. Defaults to `InMemoryQueueManager`.
            push_config_store: The `PushNotificationConfigStore` instance for managing push notification configurations. Defaults to None.
            push_sender: The `PushNotificationSender` instance for sending push notifications. Defaults to None.
            request_context_builder: The `RequestContextBuilder` instance used
              to build request contexts. Defaults to `SimpleRequestContextBuilder`.
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
