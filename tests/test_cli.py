"""Test cases for CLI interface."""

from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from semantic_scholar_mcp.cli import cli


class TestCLI:
    """Test cases for CLI commands."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Semantic Scholar MCP Server" in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

    def test_serve_command_help(self):
        """Test serve command help."""
        result = self.runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Start the MCP server" in result.output

    @patch("semantic_scholar_mcp.cli.anyio.run")
    @patch("semantic_scholar_mcp.cli.SemanticScholarServer")
    def test_serve_stdio_transport(self, mock_server_class, mock_anyio_run):
        """Test serve command with stdio transport."""
        mock_server_instance = MagicMock()
        mock_server_class.return_value = mock_server_instance

        result = self.runner.invoke(cli, ["serve", "stdio"])
        assert result.exit_code == 0
        assert "Starting Semantic Scholar MCP Server" in result.output
        mock_anyio_run.assert_called_once()

    @patch("semantic_scholar_mcp.cli.uvicorn.run")
    @patch("semantic_scholar_mcp.cli.SemanticScholarServer")
    def test_serve_http_transport(self, mock_server_class, mock_uvicorn_run):
        """Test serve command with HTTP transport."""
        mock_server_instance = MagicMock()
        mock_server_class.return_value = mock_server_instance

        result = self.runner.invoke(
            cli, ["serve", "http", "--port", "8080", "--host", "0.0.0.0"]
        )
        assert result.exit_code == 0
        assert "Starting HTTP server on http://0.0.0.0:8080" in result.output
        mock_uvicorn_run.assert_called_once()

    @patch("semantic_scholar_mcp.cli.SemanticScholarServer")
    def test_serve_with_api_key(self, mock_server_class):
        """Test serve command with API key."""
        mock_server_instance = MagicMock()
        mock_server_class.return_value = mock_server_instance

        with patch("semantic_scholar_mcp.cli.anyio.run"):
            result = self.runner.invoke(cli, ["serve", "--api-key", "test-key"])
            assert result.exit_code == 0
            assert "[OK] Semantic Scholar API key configured" in result.output
            mock_server_class.assert_called_with(api_key="test-key")

    def test_serve_without_api_key(self):
        """Test serve command without API key."""
        with (
            patch("semantic_scholar_mcp.cli.anyio.run"),
            patch("semantic_scholar_mcp.cli.SemanticScholarServer"),
        ):
            result = self.runner.invoke(cli, ["serve"])
            assert result.exit_code == 0
            assert "[WARNING] No Semantic Scholar API key found" in result.output

    def test_tools_command_help(self):
        """Test tools command help."""
        result = self.runner.invoke(cli, ["tools", "--help"])
        assert result.exit_code == 0
        assert "MCP tools for interacting with Semantic Scholar" in result.output


class TestToolsCommands:
    """Test cases for tools subcommands."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    def test_list_tools_table_format(self):
        """Test list tools command with table format."""
        result = self.runner.invoke(cli, ["tools", "list", "--format", "table"])
        assert result.exit_code == 0
        assert "Available MCP Tools" in result.output
        assert "search_paper" in result.output
        assert "get_paper" in result.output

    def test_list_tools_json_format(self):
        """Test list tools command with JSON format."""
        result = self.runner.invoke(cli, ["tools", "list", "--format", "json"])
        assert result.exit_code == 0
        # Should be valid JSON containing tool information
        assert "search_paper" in result.output
        assert '"name"' in result.output

    def test_list_tools_text_format(self):
        """Test list tools command with text format."""
        result = self.runner.invoke(cli, ["tools", "list", "--format", "text"])
        assert result.exit_code == 0
        assert "Available MCP Tools:" in result.output
        assert "search_paper" in result.output

    def test_list_tools_verbose(self):
        """Test list tools command with verbose output."""
        result = self.runner.invoke(cli, ["tools", "list", "--verbose"])
        assert result.exit_code == 0
        assert "Input Schema:" in result.output
        assert "Usage Examples:" in result.output

    @patch("semantic_scholar_mcp.cli.anyio.run")
    def test_search_paper_command(self, mock_anyio_run):
        """Test search_paper command."""

        # Mock the async function
        async def mock_search():
            pass

        mock_anyio_run.side_effect = lambda f: None

        result = self.runner.invoke(
            cli,
            [
                "tools",
                "search_paper",
                "machine learning",
                "--fields",
                "title,authors",
                "--limit",
                "5",
            ],
        )
        assert result.exit_code == 0
        mock_anyio_run.assert_called_once()

    @patch("semantic_scholar_mcp.cli.anyio.run")
    def test_get_paper_command(self, mock_anyio_run):
        """Test get_paper command."""
        # Mock the async function
        mock_anyio_run.side_effect = lambda f: None

        result = self.runner.invoke(
            cli,
            [
                "tools",
                "get_paper",
                "649def34f8be52c8b66281af98ae884c09aef38b",
                "--fields",
                "title,abstract",
            ],
        )
        assert result.exit_code == 0
        mock_anyio_run.assert_called_once()

    @patch("semantic_scholar_mcp.cli.anyio.run")
    def test_get_authors_command(self, mock_anyio_run):
        """Test get_authors command."""
        mock_anyio_run.side_effect = lambda f: None

        result = self.runner.invoke(
            cli,
            [
                "tools",
                "get_authors",
                "649def34f8be52c8b66281af98ae884c09aef38b",
                "--fields",
                "name,affiliations",
                "--limit",
                "10",
            ],
        )
        assert result.exit_code == 0
        mock_anyio_run.assert_called_once()

    @patch("semantic_scholar_mcp.cli.anyio.run")
    def test_get_citation_command(self, mock_anyio_run):
        """Test get_citation command."""
        mock_anyio_run.side_effect = lambda f: None

        result = self.runner.invoke(
            cli,
            [
                "tools",
                "get_citation",
                "649def34f8be52c8b66281af98ae884c09aef38b",
                "--format",
                "bibtex",
            ],
        )
        assert result.exit_code == 0
        mock_anyio_run.assert_called_once()

    def test_search_paper_with_filters(self):
        """Test search_paper command with all filters."""
        with patch("semantic_scholar_mcp.cli.anyio.run") as mock_anyio_run:
            mock_anyio_run.side_effect = lambda f: None

            result = self.runner.invoke(
                cli,
                [
                    "tools",
                    "search_paper",
                    "neural networks",
                    "--fields",
                    "title,authors,year",
                    "--limit",
                    "20",
                    "--offset",
                    "10",
                    "--year",
                    "2020-2023",
                    "--fields-of-study",
                    "Computer Science",
                    "--open-access-pdf",
                ],
            )
            assert result.exit_code == 0
            mock_anyio_run.assert_called_once()


class TestCLIIntegration:
    """Integration tests for CLI with mocked server responses."""

    def setup_method(self):
        """Setup test runner."""
        self.runner = CliRunner()

    @patch("semantic_scholar_mcp.cli._get_server_instance")
    @patch("semantic_scholar_mcp.cli.anyio.run")
    def test_search_paper_integration(self, mock_anyio_run, mock_get_server):
        """Test search_paper command integration with mocked server."""
        # Mock server and response
        mock_server = MagicMock()
        mock_server._handle_search_paper = AsyncMock(
            return_value=[MagicMock(text='{"data": [{"title": "Test Paper"}]}')]
        )
        mock_get_server.return_value = mock_server

        # Mock anyio.run to actually call the function
        def mock_run(coro_func):
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro_func())
            finally:
                loop.close()

        mock_anyio_run.side_effect = mock_run

        result = self.runner.invoke(cli, ["tools", "search_paper", "test query"])
        assert result.exit_code == 0
