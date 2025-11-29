"""
WebSocket Market Data Server

Provides a lightweight, real-time push channel for the JPMorgan
European Equity Dashboard (Streamlit / other frontends).

- Broadcasts STOXX 50 vs S&P 500 and any other configured tickers
- Designed to use only free data (Yahoo Finance connector)
- Can be run standalone: `python websocket_server.py`

Tech:
- asyncio
- websockets
- src.data.connectors.yahoo.YahooMarketData
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set

import websockets
from websockets.server import WebSocketServerProtocol

# App config (fallback-safe)
try:
    from config.settings import settings
except Exception:  # pragma: no cover - fallback when settings not present
    class _FallbackSettings:
        STREAMLIT_SERVER_ADDRESS = "0.0.0.0"
        WEBSOCKET_HOST = "0.0.0.0"
        WEBSOCKET_PORT = 8765
        DATA_REFRESH_INTERVAL = 5  # seconds

    settings = _FallbackSettings()  # type: ignore[assignment]

# Market data connector
from src.data.connectors.yahoo import YahooMarketData

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logger = logging.getLogger("websocket_server")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WEBSOCKET_HOST: str = getattr(settings, "WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT: int = getattr(settings, "WEBSOCKET_PORT", 8765)

# Refresh interval for market updates (seconds)
REFRESH_INTERVAL: int = max(
    2,
    int(getattr(settings, "DATA_REFRESH_INTERVAL", 5)),
)

# Default tickers to stream
DEFAULT_TICKERS: List[str] = [
    "^STOXX50E",   # Euro Stoxx 50
    "^STOXX",      # Stoxx Europe 600 (if available via Yahoo)
    "^GSPC",       # S&P 500
]


# -----------------------------------------------------------------------------
# WebSocket Server
# -----------------------------------------------------------------------------

class MarketDataWebSocketServer:
    """
    Real-time market data broadcast server.

    - Maintains a set of connected clients
    - Periodically fetches latest prices
    - Broadcasts updates to all clients as JSON
    """

    def __init__(self, host: str, port: int, tickers: List[str]) -> None:
        self.host = host
        self.port = port
        self.tickers = tickers
        self._clients: Set[WebSocketServerProtocol] = set()
        self._lock = asyncio.Lock()
        self._market_data = YahooMarketData()
        self._server: websockets.server.Serve | None = None
        self._broadcast_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    async def register(self, ws: WebSocketServerProtocol) -> None:
        async with self._lock:
            self._clients.add(ws)
        logger.info("Client connected: %s | total=%d", ws.remote_address, len(self._clients))

    async def unregister(self, ws: WebSocketServerProtocol) -> None:
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)
        logger.info("Client disconnected: %s | total=%d", ws.remote_address, len(self._clients))

    # ------------------------------------------------------------------
    # Broadcast loop
    # ------------------------------------------------------------------

    async def _broadcast_loop(self) -> None:
        """
        Periodically fetch market data and broadcast to all clients.
        """
        logger.info("Starting broadcast loop (interval=%ss)", REFRESH_INTERVAL)

        while True:
            try:
                if not self._clients:
                    await asyncio.sleep(REFRESH_INTERVAL)
                    continue

                # Fetch latest prices
                prices: Dict[str, Dict[str, float]] = self._market_data.get_realtime_prices(
                    tickers=self.tickers
                )

                payload = {
                    "type": "market_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": prices,
                }

                message = json.dumps(payload)

                # Send to all connected clients
                async with self._lock:
                    dead_clients: List[WebSocketServerProtocol] = []
                    for ws in self._clients:
                        try:
                            await ws.send(message)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to send to client %s: %s", ws.remote_address, exc)
                            dead_clients.append(ws)

                    # Clean up dead clients
                    for ws in dead_clients:
                        self._clients.discard(ws)

            except Exception as exc:  # noqa: BLE001
                logger.error("Error in broadcast loop: %s", exc, exc_info=True)

            await asyncio.sleep(REFRESH_INTERVAL)

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def handler(self, ws: WebSocketServerProtocol) -> None:
        """
        Handle an incoming WebSocket connection.

        Protocol:
        - Server pushes market_update frames periodically
        - Client may optionally send:
            { "type": "ping" }   -> server replies with pong
            { "type": "subscribe", "tickers": [...] } -> override tickers (per connection in future)
        """
        await self.register(ws)

        try:
            # Send immediate snapshot
            try:
                prices = self._market_data.get_realtime_prices(tickers=self.tickers)
                await ws.send(
                    json.dumps(
                        {
                            "type": "market_snapshot",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": prices,
                        }
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send initial snapshot: %s", exc)

            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    # Ignore unparseable messages
                    continue

                msg_type = data.get("type")

                if msg_type == "ping":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "pong",
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )
                elif msg_type == "subscribe":
                    # For now, this is global; later can be per-client
                    new_tickers = data.get("tickers") or []
                    if isinstance(new_tickers, list) and new_tickers:
                        logger.info("Updating tickers to: %s", new_tickers)
                        self.tickers = new_tickers
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "subscribe_ack",
                                    "tickers": self.tickers,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )
                        )
                else:
                    # Unknown message types can be ignored or logged
                    logger.debug("Unknown message from client: %s", data)

        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosedError:
            pass
        finally:
            await self.unregister(ws)

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        logger.info("Starting WebSocket server on %s:%d", self.host, self.port)
        self._server = await websockets.serve(self.handler, self.host, self.port)

        # Launch broadcast loop
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        logger.info("Stopping WebSocket server...")
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

        logger.info("WebSocket server stopped.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

async def main() -> None:
    server = MarketDataWebSocketServer(
        host=WEBSOCKET_HOST,
        port=WEBSOCKET_PORT,
        tickers=DEFAULT_TICKERS,
    )
    await server.start()

    # Keep running until Ctrl+C
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Received shutdown signal.")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
