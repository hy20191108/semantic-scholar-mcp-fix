"""Integration tests and edge cases."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from mcp.types import TextContent

from semantic_scholar_mcp.server import SemanticScholarServer


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.anyio
    async def test_search_paper_with_all_parameters(
        self, server_without_api_key: SemanticScholarServer
    ):
        """Test search with all possible parameters."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [], "total": 0}
            mock_to_thread.return_value = mock_response

            args = {
                "query": "machine learning",
                "fields": "title,abstract,authors",
                "publicationTypes": "JournalArticle,Conference",
                "minCitationCount": 10,
                "publicationDateOrYear": "2020-01-01:2023-12-31",
                "year": "2020-2023",
                "venue": "Nature,Science",
                "fieldsOfStudy": "Computer Science,Mathematics",
                "openAccessPdf": True,
                "offset": 20,
                "limit": 50,
            }

            result = await server_without_api_key._handle_search_paper(args)
            assert len(result) == 1
            assert isinstance(result[0], TextContent)

            # Verify API call was made with correct parameters
            mock_to_thread.assert_called_once()
            call_args = mock_to_thread.call_args
            assert call_args[1]["params"]["query"] == "machine learning"
            assert (
                call_args[1]["params"]["openAccessPdf"] == ""
            )  # Should be empty string when True

    @pytest.mark.anyio
    async def test_search_paper_limit_capping(
        self, server_without_api_key: SemanticScholarServer
    ):
        """Test that search limit is capped at 100."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_to_thread.return_value = mock_response

            args = {"query": "test", "limit": 200}  # Over the limit
            await server_without_api_key._handle_search_paper(args)

            # Check that limit was capped at 100
            call_args = mock_to_thread.call_args
            assert call_args[1]["params"]["limit"] == 100

    @pytest.mark.anyio
    async def test_get_authors_limit_capping(
        self, server_without_api_key: SemanticScholarServer
    ):
        """Test that get_authors limit is capped at 1000."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_to_thread.return_value = mock_response

            args = {"paper_id": "test_id", "limit": 2000}  # Over the limit
            await server_without_api_key._handle_get_authors(args)

            # Check that limit was capped at 1000
            call_args = mock_to_thread.call_args
            assert call_args[1]["params"]["limit"] == 1000

    @pytest.mark.anyio
    async def test_handle_get_citation_no_citation_styles(
        self, server_without_api_key: SemanticScholarServer
    ):
        """Test get_citation with no citation styles available."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "paperId": "test",
                "abstract": "test",
            }  # No citationStyles
            mock_to_thread.return_value = mock_response

            args = {"paper_id": "test_id", "format": "bibtex"}
            result = await server_without_api_key._handle_get_citation(args)

            assert len(result) == 1
            assert "No citation styles available" in result[0].text

    @pytest.mark.anyio
    async def test_network_timeout_simulation(
        self, server_without_api_key: SemanticScholarServer
    ):
        """Test handling of network timeouts."""
        with patch(
            "asyncio.to_thread", side_effect=asyncio.TimeoutError("Request timed out")
        ):
            args = {"query": "test"}
            result = await server_without_api_key._handle_search_paper(args)

            assert len(result) == 1
            assert "Error searching papers" in result[0].text

    @pytest.mark.anyio
    async def test_malformed_json_response(
        self, server_without_api_key: SemanticScholarServer
    ):
        """Test handling of malformed JSON responses."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_to_thread.return_value = mock_response

            args = {"query": "test"}
            result = await server_without_api_key._handle_search_paper(args)

            assert len(result) == 1
            assert "Error searching papers" in result[0].text

    def test_default_field_values(self, server_without_api_key: SemanticScholarServer):
        """Test that default field values are correctly set."""
        assert hasattr(server_without_api_key, "search_paper_default_fields")
        assert hasattr(server_without_api_key, "get_paper_default_fields")
        assert hasattr(server_without_api_key, "get_authors_default_fields")

        # Verify they contain expected fields
        assert "paperId" in server_without_api_key.search_paper_default_fields
        assert "title" in server_without_api_key.search_paper_default_fields
        assert "authorId" in server_without_api_key.get_authors_default_fields

    @pytest.mark.anyio
    async def test_concurrent_requests(
        self, server_without_api_key: SemanticScholarServer
    ):
        """Test handling multiple concurrent requests."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_to_thread.return_value = mock_response

            # Make multiple concurrent requests
            tasks = []
            for i in range(5):
                args = {"query": f"test query {i}"}
                task = server_without_api_key._handle_search_paper(args)
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # All requests should succeed
            assert len(results) == 5
            for result in results:
                assert len(result) == 1
                assert isinstance(result[0], TextContent)

    @pytest.mark.anyio
    async def test_api_rate_limiting_response(
        self, server_without_api_key: SemanticScholarServer
    ):
        """Test handling of API rate limiting (429 status)."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.text = "Rate limit exceeded"
            mock_to_thread.return_value = mock_response

            args = {"query": "test"}
            result = await server_without_api_key._handle_search_paper(args)

            assert len(result) == 1
            assert "Error: API returned status 429" in result[0].text
            assert "Rate limit exceeded" in result[0].text

    @pytest.mark.anyio
    async def test_empty_query_search(
        self, server_without_api_key: SemanticScholarServer
    ):
        """Test search with empty query."""
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request: Empty query"
            mock_to_thread.return_value = mock_response

            args = {"query": ""}
            result = await server_without_api_key._handle_search_paper(args)

            assert len(result) == 1
            assert "Error: API returned status 400" in result[0].text

    @pytest.mark.anyio
    async def test_invalid_paper_id_formats(
        self, server_without_api_key: SemanticScholarServer
    ):
        """Test various invalid paper ID formats."""
        invalid_ids = [
            "",  # Empty
            "invalid-format",  # Invalid format
            "DOI:",  # Incomplete DOI
            "ARXIV:",  # Incomplete ArXiv
        ]

        for paper_id in invalid_ids:
            with patch("asyncio.to_thread") as mock_to_thread:
                mock_response = MagicMock()
                mock_response.status_code = 404
                mock_response.text = "Not Found"
                mock_to_thread.return_value = mock_response

                args = {"paper_id": paper_id}
                result = await server_without_api_key._handle_get_paper(args)

                assert len(result) == 1
                assert f"Paper not found: {paper_id}" in result[0].text


class TestAPIKeyHandling:
    """Test API key handling in different scenarios."""

    def test_headers_with_different_api_keys(self):
        """Test header generation with different API key formats."""
        # Test with normal API key
        server1 = SemanticScholarServer(api_key="sk-test-key")
        headers1 = server1._get_headers()
        assert headers1["x-api-key"] == "sk-test-key"

        # Test with empty string API key
        server2 = SemanticScholarServer(api_key="")
        headers2 = server2._get_headers()
        assert "x-api-key" not in headers2

        # Test with None API key
        server3 = SemanticScholarServer(api_key=None)
        headers3 = server3._get_headers()
        assert "x-api-key" not in headers3

    @pytest.mark.anyio
    async def test_api_calls_with_and_without_key(self):
        """Test that API calls include key when provided."""
        # With API key
        server_with_key = SemanticScholarServer(api_key="test-key")

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_to_thread.return_value = mock_response

            args = {"query": "test"}
            await server_with_key._handle_search_paper(args)

            # Check that API key was included in headers
            call_args = mock_to_thread.call_args
            headers = call_args[1]["headers"]
            assert headers["x-api-key"] == "test-key"

        # Without API key
        server_without_key = SemanticScholarServer()

        with patch("asyncio.to_thread") as mock_to_thread:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_to_thread.return_value = mock_response

            args = {"query": "test"}
            await server_without_key._handle_search_paper(args)

            # Check that API key was not included in headers
            call_args = mock_to_thread.call_args
            headers = call_args[1]["headers"]
            assert "x-api-key" not in headers
