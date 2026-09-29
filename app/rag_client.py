from typing import Any

import httpx

from .config import RAG_BASE_URL


class RAGClient:

    def __init__(self):
        self.url = (
            f"{RAG_BASE_URL}"
            "/api/v1/chat/retrieve"
        )

    async def retrieve(
        self,
        query: str,
    ) -> list[dict[str, Any]]:

        payload = {
            "query": query
        }

        async with httpx.AsyncClient(
            timeout=60.0
        ) as client:

            response = await client.post(
                self.url,
                json=payload,
            )

        if response.status_code != 200:

            raise RuntimeError(
                "RAG server retrieval failed.\n"
                f"Status: {response.status_code}\n"
                f"Response: {response.text}"
            )

        data = response.json()

        if isinstance(data, dict):

            return data.get(
                "evidence",
                []
            )

        if isinstance(data, list):

            return data

        return []