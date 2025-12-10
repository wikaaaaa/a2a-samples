import json
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.client.errors import A2AClientJSONRPCError
from a2a.client.middleware import ClientCallContext
from a2a.client.transports import ClientTransport
from a2a.extensions.common import find_extension_by_uri
from a2a.types import (AgentCapabilities, AgentCard, AgentExtension, Artifact,
                       ContentTypeNotSupportedError, DataPart,
                       DeleteTaskPushNotificationConfigParams, FilePart,
                       FileWithBytes, FileWithUri,
                       GetTaskPushNotificationConfigParams, JSONRPCError,
                       JSONRPCErrorResponse,
                       ListTaskPushNotificationConfigParams, Message,
                       MessageSendParams, Part, Role, Task,
                       TaskArtifactUpdateEvent, TaskIdParams,
                       TaskPushNotificationConfig, TaskQueryParams, TaskState,
                       TaskStatus, TaskStatusUpdateEvent, TextPart,
                       UnsupportedOperationError)

# --- Mapping Functions (Dict-based) ---


def _part_to_content(part: Part) -> dict[str, Any]:
    """Maps an A2A Part object to an Interactions API Content dictionary."""
    if isinstance(part.root, TextPart):
        content = {'text': part.root.text}
        if part.root.metadata and (annotations := part.root.metadata.get('annotations')):
            content['annotations'] = annotations
        return {'text': content}
    elif isinstance(part.root, FilePart):
        mime_type = None
        blob_content = {}

        if isinstance(part.root.file, FileWithBytes):
            blob_content['data'] = part.root.file.bytes
            mime_type = part.root.file.mime_type
        elif isinstance(part.root.file, FileWithUri):
            blob_content['uri'] = part.root.file.uri
            mime_type = part.root.file.mime_type

        if part.root.metadata:
            resolution = part.root.metadata.get('resolution')
            if resolution:
                blob_content['resolution'] = resolution

        if mime_type:
            blob_content['mime_type'] = mime_type
            if mime_type.startswith('image'):
                return {'image': blob_content}
            elif mime_type.startswith('audio'):
                return {'audio': blob_content}
            elif mime_type.startswith('application/pdf'):
                return {'document': blob_content}
            elif mime_type.startswith('video'):
                return {'video': blob_content}
            else:
                raise A2AClientJSONRPCError(
                    JSONRPCErrorResponse(
                        error=ContentTypeNotSupportedError(
                            message=f'Unsupported file MIME type for Interactions API mapping: {mime_type}'
                        )
                    )
                )

    elif isinstance(part.root, DataPart):
        return {'text': {'text': json.dumps(part.root.data)}}

    raise A2AClientJSONRPCError(
        JSONRPCErrorResponse(
            error=ContentTypeNotSupportedError(
                message=f'Unsupported A2A Part type for Interactions API mapping: {type(part.root)}'
            )
        )
    )


def _a2a_request_to_interaction(message_params: MessageSendParams, agent_interaction: dict[str, Any]) -> dict[str, Any]:
    """Maps an A2A Message to the input structure for creating an Interaction."""
    message = message_params.message
    interaction_data = {
        'contentList': {'contents': [_part_to_content(p) for p in message.parts]},
        'agentInteraction': agent_interaction,
    }
    if message_params.configuration:
        interaction_data['background'] = not message_params.configuration.blocking
    if message.task_id:
        interaction_data['previousInteractionId'] = message.task_id

    return interaction_data


def _interaction_status_to_a2a_task_state(status_str: str) -> TaskState:
    """Maps Interactions API status string to A2A TaskState."""
    mapping = {
        'UNSPECIFIED': TaskState.unknown,
        'IN_PROGRESS': TaskState.working,
        'REQUIRES_ACTION': TaskState.input_required,
        'COMPLETED': TaskState.completed,
        'FAILED': TaskState.failed,
        'CANCELLED': TaskState.canceled,
    }
    return mapping.get(status_str, TaskState.unknown)


def _thought_to_message(thought_content: dict[str, Any]) -> Message:
    """Maps an Interactions API thought content to an A2A Message."""
    thought_parts: list[Part] = []
    summary = thought_content.get('summary', {})
    items = summary.get('items', [])
    for item in items:
        thought_parts.append(_content_to_part(item))

    if not thought_parts:
        raise A2AClientJSONRPCError(
            JSONRPCErrorResponse(
                error=ContentTypeNotSupportedError(message=f'Unsupported or empty thought content: {thought_content}')
            )
        )
    return Message(message_id=str(uuid.uuid4()), role=Role.agent, parts=thought_parts)


def _content_to_part(
    content: dict[str, Any],
) -> Part:
    """Maps an Interactions API Content dictionary to an A2A Part object (excluding thoughts)."""
    if 'text' in content:
        text_data = content['text']
        return Part(
            root=TextPart(
                text=text_data.get('text', ''),
                metadata={'annotations': text_data.get('annotations')} if text_data.get('annotations') else {},
            )
        )

    for blob_type in ['image', 'audio', 'document', 'video']:
        if blob_type in content:
            blob_data = content[blob_type]

            mime_type = blob_data.get('mime_type') or f'{blob_type}/unknown'
            if blob_type == 'document' and not blob_data.get('mime_type'):
                mime_type = 'application/octet-stream'

            metadata = {}
            if 'resolution' in blob_data:
                metadata['resolution'] = blob_data['resolution']

            if 'data' in blob_data:
                return Part(
                    root=FilePart(
                        file=FileWithBytes(bytes=blob_data['data'], mime_type=mime_type),
                        metadata=metadata if metadata else None,
                    )
                )
            elif 'uri' in blob_data:
                return Part(
                    root=FilePart(
                        file=FileWithUri(uri=blob_data['uri'], mime_type=mime_type),
                        metadata=metadata if metadata else None,
                    )
                )

    if 'function' in content:
        return Part(root=DataPart(data={'function_call': content['function']}))

    if 'functionResponse' in content:
        return Part(root=DataPart(data={'function_response': content['functionResponse']}))

    raise A2AClientJSONRPCError(
        JSONRPCErrorResponse(
            error=ContentTypeNotSupportedError(
                message=f'Unsupported Interactions API Content type: {list(content.keys())}'
            )
        )
    )


def convert_interaction_to_task(interaction: dict[str, Any]) -> Task:
    """Converts a raw Interactions API dictionary response to an A2A Task."""
    task_id = interaction.get('id', '')

    status_obj = interaction.get('status', {})
    status_str = status_obj.get('status', 'UNSPECIFIED')
    task_status_state = _interaction_status_to_a2a_task_state(status_str)

    a2a_history = []
    turn_list = interaction.get('turnList', {}).get('turns', [])

    if 'content' in interaction:
        a2a_history.append(
            Message(
                message_id='request', role=Role.user, parts=[_content_to_part(interaction['content'])], task_id=task_id
            )
        )
    elif 'stringContent' in interaction:
        a2a_history.append(
            Message(
                message_id='request',
                role=Role.user,
                parts=[Part(root=TextPart(text=interaction['stringContent']))],
                task_id=task_id,
            )
        )
    elif 'contentList' in interaction:
        contents = interaction['contentList'].get('contents', [])
        a2a_history.append(
            Message(
                message_id='request',
                role=Role.user,
                parts=[_content_to_part(content) for content in contents],
                task_id=task_id,
            )
        )

    for i, turn in enumerate(turn_list):
        turn_role = Role.user if turn.get('role') == 'user' else Role.agent
        a2a_message_parts = []

        content_list = turn.get('contentList', {}).get('contents', [])
        if content_list:
            for content in content_list:
                try:
                    a2a_message_parts.append(_content_to_part(content))
                except A2AClientJSONRPCError as e:
                    print(f'Warning: Skipping unsupported history content: {e}')
                    continue
        elif turn.get('contentString'):
            a2a_message_parts.append(Part(root=TextPart(text=turn['contentString'])))

        if a2a_message_parts:
            a2a_history.append(
                Message(
                    message_id=f'{task_id}_turn_{i}',
                    role=turn_role,
                    parts=a2a_message_parts,
                    task_id=task_id,
                )
            )

    a2a_artifacts = []
    thought_messages = []
    outputs = interaction.get('outputs', [])

    for i, output_content in enumerate(outputs):
        if 'thought' in output_content:
            try:
                thought_msg = _thought_to_message(output_content['thought'])
                thought_msg.message_id = f'{task_id}_thought_{i}'
                thought_msg.task_id = task_id
                thought_messages.append(thought_msg)
            except A2AClientJSONRPCError:
                continue
        else:
            try:
                a2a_part = _content_to_part(output_content)
                a2a_artifacts.append(
                    Artifact(
                        artifact_id=f'{task_id}_output_{i}',
                        parts=[a2a_part],
                        name=f'output-{i}',
                        metadata={},
                    )
                )
            except A2AClientJSONRPCError as e:
                print(f'Warning: Skipping unsupported content type: {e}')
                continue

    # Append thoughts to history
    a2a_history.extend(thought_messages)

    # Determine status message
    task_status_message: Message | None = None
    if thought_messages:
        task_status_message = thought_messages[-1]

    # Error handling overrides thought status
    error_obj = interaction.get('error') or status_obj.get('error')
    if error_obj:
        task_status_message = Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[Part(root=TextPart(text=json.dumps(error_obj)))],
        )

    return Task(
        id=task_id,
        context_id=task_id,
        status=TaskStatus(state=task_status_state, message=task_status_message),
        artifacts=a2a_artifacts if a2a_artifacts else None,
        history=a2a_history if a2a_history else None,
    )


class InteractionsApiTransport(ClientTransport):
    INTERACTIONS_API_VERSION = 'v1beta'
    EXTENSION_URI = 'https://generativelanguage.googleapis.com/v1beta/a2a'
    TRANSPORT_NAME = 'interactions-api'

    _WELL_KNOWN_AGENTS_INTERACTION_CONFIG: dict[str, dict[str, Any]] = {
        'deep-research-preview': {
            'deep_research_config': {
                'thinking_summaries': 'THINK_SUMMARIES_AUTO',
            },
        },
    }
    _WELL_KNOWN_AGENTS_NAMES: dict[str, str] = {
        'deep-research-preview': 'Deep Research Preview',
    }
    _WELL_KNOWN_AGENTS_DESCRIPTIONS: dict[str, str] = {
        'deep-research-preview': 'Agent powered by Google Deep Research Preview',
    }
    _WELL_KNOWN_AGENTS_SKILLS: dict[str, list[str]] = {}

    def __init__(self, card: AgentCard, api_key: str | None = None):
        self._card = card
        self._api_key = api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')

        if not self._api_key:
            raise ValueError(
                'API Key not provided and not found in environment variables (GOOGLE_API_KEY or GEMINI_API_KEY).'
            )

        self._base_url = self._card.url
        self._client = httpx.AsyncClient(base_url=self._base_url)
        self._client.headers['x-goog-api-key'] = self._api_key

        # Extract agent_interaction from AgentCard extension
        extension = find_extension_by_uri(self._card, self.EXTENSION_URI)
        if not extension or not extension.params:
            raise ValueError(
                'invalid AgentCard: missing required extension; use make_card to create a correctly formatted AgentCard'
            )
        self._agent_interaction: dict[str, Any] = extension.params['agentInteraction']

    @classmethod
    def make_card(
        cls,
        url: str,
        agent_name: str,
        agent_config: dict[str, Any] | None = None,
        agent_card_overrides: dict[str, Any] | None = None,
    ) -> AgentCard:
        # Start with well-known agent config if applicable
        final_agent_config = cls._WELL_KNOWN_AGENTS_INTERACTION_CONFIG.get(agent_name, {}).copy()
        if agent_config:
            final_agent_config.update(agent_config)

        agent_interaction: dict[str, Any] = {
            'agent': agent_name,
        }
        if agent_name == 'deep-research-preview':
            agent_interaction['deepResearchConfig'] = agent_config
        else:
            agent_interaction['dynamicConfig'] = agent_config

        # Default AgentCard values
        card_defaults: dict[str, Any] = {
            'url': url,
            'preferred_transport': 'interactions-api',
            'capabilities': AgentCapabilities(
                streaming=True,
                extensions=[
                    AgentExtension(
                        uri=cls.EXTENSION_URI,
                        required=True,
                        params={
                            'agentInteraction': agent_interaction,
                        },
                    )
                ],
            ),
            'name': cls._WELL_KNOWN_AGENTS_NAMES.get(agent_name, agent_name),
            'description': cls._WELL_KNOWN_AGENTS_DESCRIPTIONS.get(
                agent_name, 'Agent powered by Google Interactions API'
            ),
            'default_input_modes': ['text/plain'],
            'default_output_modes': ['text/plain'],
            'skills': cls._WELL_KNOWN_AGENTS_SKILLS.get(agent_name, []),
            'version': '0.1.0',
        }

        # Apply overrides
        if agent_card_overrides:
            card_defaults.update(agent_card_overrides)

        return AgentCard(**card_defaults)

    @classmethod
    def setup(cls, client_config: ClientConfig, client_factory: ClientFactory, api_key: str | None = None):
        client_config.supported_transports.append(cls.TRANSPORT_NAME)
        client_factory.register(cls.TRANSPORT_NAME, lambda card, url, config, interceptors: cls(card, api_key))

    async def _make_request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> httpx.Response:
        try:
            headers = {'Content-Type': 'application/json'}
            if stream:
                headers['Accept'] = 'text/event-stream'

            if method == 'POST':
                response = await self._client.post(path, json=json_data, params=params, headers=headers, timeout=None)
            elif method == 'GET':
                response = await self._client.get(path, params=params, headers=headers, timeout=None)
            else:
                raise ValueError(f'Unsupported HTTP method: {method}')

            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                raise A2AClientJSONRPCError(
                    JSONRPCErrorResponse(
                        error=JSONRPCError(code=-32602, message='Invalid parameters', data=e.response.json())
                    )
                ) from e
            elif e.response.status_code == 404:
                raise A2AClientJSONRPCError(
                    JSONRPCErrorResponse(
                        error=JSONRPCError(code=-32001, message='Task not found', data=e.response.json())
                    )
                ) from e
            elif e.response.status_code == 405:
                raise A2AClientJSONRPCError(
                    JSONRPCErrorResponse(
                        error=UnsupportedOperationError(message='Method not supported by Interactions API transport.')
                    )
                ) from e
            else:
                raise A2AClientJSONRPCError(
                    JSONRPCErrorResponse(
                        error=JSONRPCError(code=-32603, message='Internal error', data=e.response.json())
                    )
                ) from e
        except httpx.RequestError as e:
            raise A2AClientJSONRPCError(
                JSONRPCErrorResponse(error=JSONRPCError(code=-32603, message=f'Network or client error: {e}'))
            ) from e

    async def send_message(
        self,
        request: MessageSendParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task | Message:
        interaction_input = _a2a_request_to_interaction(request, self._agent_interaction)
        payload = {
            'stream': False,
            'store': True,
            'interaction': interaction_input['interaction'],
        }
        # Add root-level params if present in extension config
        # Add root-level params if present in extension config
        if agent := self._agent_interaction.get('agent'):
            payload['agent'] = agent
        if deep_research_config := self._agent_interaction.get('deep_research_config'):
            payload['deepResearchConfig'] = deep_research_config
        if dynamic_config := self._agent_interaction.get('dynamic_config'):
            payload['dynamicConfig'] = dynamic_config

        response = await self._make_request(
            'POST', f'/{self.INTERACTIONS_API_VERSION}/interactions:create', json_data=payload
        )
        return convert_interaction_to_task(response.json())

    async def _process_stream(
        self,
        response: httpx.Response,
        initial_task: Task | None = None,
    ) -> AsyncGenerator[Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent]:
        current_task = initial_task
        async for line in response.aiter_lines():
            if line.startswith('data:'):
                try:
                    json_data = json.loads(line[len('data:') :].strip())
                    event_type = json_data.get('eventType')
                    event_data = json_data.get('event', {})

                    if not event_type:
                        continue

                    if event_type == 'InteractionEvent':
                        if 'interaction' in event_data:
                            current_task = convert_interaction_to_task(event_data['interaction'])
                            self._streaming_tasks[current_task.id] = current_task
                            yield current_task
                        elif event_data.get('completed') and current_task:
                            current_task.status.state = TaskState.completed
                            yield TaskStatusUpdateEvent(
                                task_id=current_task.id,
                                context_id=current_task.context_id,
                                status=current_task.status,
                                final=True,
                            )
                            if current_task.id in self._streaming_tasks:
                                del self._streaming_tasks[current_task.id]

                    elif event_type == 'InteractionStatusUpdate' and current_task:
                        status_str = event_data.get('status', 'UNSPECIFIED')
                        current_task.status.state = _interaction_status_to_a2a_task_state(status_str)

                        error_data = event_data.get('error')
                        if error_data:
                            current_task.status.message = Message(
                                message_id=str(uuid.uuid4()),
                                role=Role.agent,
                                parts=[Part(root=TextPart(text=json.dumps(error_data)))],
                            )

                        yield TaskStatusUpdateEvent(
                            task_id=current_task.id,
                            context_id=current_task.context_id,
                            status=current_task.status,
                            final=False,
                        )

                    elif event_type == 'Content' and current_task:
                        if 'thought' in event_data:
                            try:
                                # Convert thought content directly to an A2A Message
                                thought_message = _map_interaction_thought_to_a2a_message(event_data['thought'])
                                current_task.status.message = thought_message
                                yield TaskStatusUpdateEvent(
                                    task_id=current_task.id,
                                    context_id=current_task.context_id,
                                    status=current_task.status,
                                    final=False,
                                )
                            except A2AClientJSONRPCError as e:
                                print(f'Warning: Skipping unsupported thought content in stream: {e}')
                        else:
                            try:
                                a2a_part = _content_to_part(event_data)
                                yield TaskArtifactUpdateEvent(
                                    task_id=current_task.id,
                                    context_id=current_task.context_id,
                                    artifact=Artifact(
                                        artifact_id=str(uuid.uuid4()),
                                        parts=[a2a_part],
                                        name='artifact',
                                        metadata={},
                                    ),
                                    last_chunk=False,
                                )
                            except A2AClientJSONRPCError as e:
                                print(f'Warning: Skipping unsupported content in stream: {e}')

                except json.JSONDecodeError:
                    print(f'Warning: JSON decode error: {line}')

            elif line.strip() == 'event: end' and current_task:
                if current_task.id in self._streaming_tasks:
                    del self._streaming_tasks[current_task.id]
                yield TaskStatusUpdateEvent(
                    task_id=current_task.id,
                    context_id=current_task.context_id,
                    status=current_task.status,
                    final=True,
                )

    async def send_message_streaming(
        self,
        request: MessageSendParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncGenerator[Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent]:
        interaction_input = _a2a_request_to_interaction(request.message, self._agent_interaction)
        payload = {
            'stream': True,
            'store': True,
            'interaction': interaction_input['interaction'],
        }
        # Add root-level params
        # Add root-level params if present in extension config
        if agent := self._agent_interaction.get('agent'):
            payload['agent'] = agent
        if deep_research_config := self._agent_interaction.get('deep_research_config'):
            payload['deepResearchConfig'] = deep_research_config
        if dynamic_config := self._agent_interaction.get('dynamic_config'):
            payload['dynamicConfig'] = dynamic_config

        async with self._client.stream(
            'POST',
            f'/{self.INTERACTIONS_API_VERSION}/interactions:createStream',
            json=payload,
            headers={'Accept': 'text/event-stream'},
            timeout=None,
        ) as response:
            response.raise_for_status()
            async for event in self._process_stream(response):
                yield event

    async def get_task(
        self,
        request: TaskQueryParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task:
        response = await self._make_request('GET', f'/{self.INTERACTIONS_API_VERSION}/interactions/{request.id}:poll')
        return convert_interaction_to_task(response.json())

    async def cancel_task(
        self,
        request: TaskIdParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> Task:
        response = await self._make_request(
            'POST', f'/{self.INTERACTIONS_API_VERSION}/interactions/{request.id}:cancel'
        )
        return convert_interaction_to_task(response.json())

    async def set_task_callback(
        self,
        request: TaskPushNotificationConfig,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> TaskPushNotificationConfig:
        raise A2AClientJSONRPCError(
            JSONRPCErrorResponse(
                error=UnsupportedOperationError(
                    message='Push Notifications are not supported by Interactions API transport.'
                )
            )
        )

    async def get_task_callback(
        self,
        request: GetTaskPushNotificationConfigParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> TaskPushNotificationConfig:
        raise A2AClientJSONRPCError(
            JSONRPCErrorResponse(
                error=UnsupportedOperationError(
                    message='Push Notifications are not supported by Interactions API transport.'
                )
            )
        )

    async def list_task_callback(
        self,
        request: ListTaskPushNotificationConfigParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> list[TaskPushNotificationConfig]:
        raise A2AClientJSONRPCError(
            JSONRPCErrorResponse(
                error=UnsupportedOperationError(
                    message='Push Notifications are not supported by Interactions API transport.'
                )
            )
        )
  
    async def delete_task_callback(
        self,
        request: DeleteTaskPushNotificationConfigParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> None:
        raise A2AClientJSONRPCError(
            JSONRPCErrorResponse(
                error=UnsupportedOperationError(
                    message='Push Notifications are not supported by Interactions API transport.'
                )
            )
        )
    

    async def resubscribe(
        self,
        request: TaskIdParams,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> AsyncGenerator[Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent]:
        async with self._client.stream(
            'GET',
            f'/{self.INTERACTIONS_API_VERSION}/interactions/{request.id}:stream',
            headers={'Accept': 'text/event-stream'},
            timeout=None,
        ) as response:
            response.raise_for_status()
            current_task: Task | None = self._streaming_tasks.get(request.id)
            if not current_task:
                try:
                    # Sync state first
                    current_task = await self.get_task(TaskQueryParams(id=request.id))
                    self._streaming_tasks[request.id] = current_task
                    yield current_task
                except Exception as e:
                    print(f'Error fetching task for resubscribe: {e}')
                    raise

            async for event in self._process_stream(response, initial_task=current_task):
                yield event

    async def get_card(
        self,
        *,
        context: ClientCallContext | None = None,
        extensions: list[str] | None = None,
    ) -> AgentCard:
        return self._card

    async def close(self) -> None:
        await self._client.aclose()
