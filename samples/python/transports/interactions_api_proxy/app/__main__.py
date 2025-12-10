import logging
import os
import sys

import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (BasePushNotificationSender,
                              InMemoryPushNotificationConfigStore,
                              InMemoryTaskStore)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from app.agent import CurrencyAgent
from app.agent_executor import CurrencyAgentExecutor
from app.request_handler import InteractionsAPIProxyRequestHandler
from dotenv import load_dotenv
from interactions_api.interactions_api_transport import \
    InteractionsApiTransport

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10000)
def main(host, port):
    """Starts the Interactions API proxy server."""
    try:

        if not os.getenv('GOOGLE_API_KEY') and not os.getenv('GEMINI_API_KEY'):
            raise MissingAPIKeyError(
                'both GOOGLE_API_KEY and GEMINI_API_KEY environment variables not set.'
            )

        interactions_agent_card = InteractionsApiTransport.make_card(
            url='https://generativelanguage.googleapis.com/v1beta/interactions',
            agent="interactions-api",
        )

        interaction_api_transport_object = InteractionsApiTransport(
            card=interactions_agent_card
        )

        request_handler = InteractionsAPIProxyRequestHandler(
            interactions_api_transport=interaction_api_transport_object
        )

        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id='interactions_api_proxy',
            name='Interactions API proxy',
            description='Serves as a proxy between A2A and Interactions API',
            tags=['interactions API proxy'],
        )

        agent_card = AgentCard(
            name='Interactions API proxy Agent',
            description='Serves as a proxy between A2A and Interactions API',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            default_input_modes=['text/plain'],
            default_output_modes=['text/plain'],
            capabilities=capabilities,
            skills=[skill],
        )

        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
