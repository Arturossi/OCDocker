#!/usr/bin/env python3

# Description
###############################################################################
"""Browser characterization tests for the OCScore Workbench application."""

# Imports
###############################################################################
from __future__ import annotations

import json
import os
import re
import socket
import threading
import time

from collections.abc import Iterator
from pathlib import Path

import pytest
import uvicorn

from OCDocker.Workbench import build_workbench_api_app

playwright = pytest.importorskip("playwright.sync_api")

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Constants
###############################################################################

PLOTLY_STUB = """
(() => {
  function render(host, data, layout, config) {
    host.data = data;
    host.layout = layout;
    host.config = config;
    host.__plotlyHandlers = host.__plotlyHandlers || {};
    host.on = (name, callback) => {
      host.__plotlyHandlers[name] = callback;
      return host;
    };
    host.replaceChildren(Object.assign(document.createElement("span"), {
      className: "plotly-characterization-stub",
      textContent: `Plotly stub: ${(data || []).length} trace(s)`,
    }));
    return Promise.resolve(host);
  }

  window.Plotly = {
    newPlot: render,
    react: render,
    purge(host) {
      delete host.data;
      delete host.layout;
      host.replaceChildren();
    },
    Plots: { resize() {} },
    toImage() {
      return Promise.resolve("data:image/png;base64,");
    },
  };
})();
"""


# Fixtures
###############################################################################


@pytest.fixture(scope="session")
def browser() -> Iterator[object]:
    """Launch one headless Chromium instance for this test module."""

    with playwright.sync_playwright() as manager:
        launch_options: dict[str, object] = {"headless": True}
        executable = os.environ.get("OCDOCKER_PLAYWRIGHT_EXECUTABLE")
        if executable:
            launch_options["executable_path"] = executable
        instance = manager.chromium.launch(**launch_options)
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def page(browser: object) -> Iterator[object]:
    """Create an isolated browser context for one characterization test."""

    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    current_page = context.new_page()
    try:
        yield current_page
    finally:
        context.close()


@pytest.fixture
def ocscore_root(tmp_path: Path) -> Path:
    """Create a minimal completed baseline and one ablation study."""

    replica = tmp_path / "replica_1"
    replica.mkdir()
    (replica / "metrics.csv").write_text("metric,value\nBEDROC,0.77\n", encoding="utf-8")
    for dataset, marker in (("pdbbind", "pdbbind_best.pt"), ("dudez", "dudez_best.pt")):
        target = replica / dataset
        target.mkdir()
        (target / marker).write_bytes(b"model")

    ablation = tmp_path / "ablation" / "no_ligand" / "replica_1"
    ablation.mkdir(parents=True)
    (ablation / "metrics.csv").write_text("metric,value\nBEDROC,0.52\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def workbench_url(ocscore_root: Path) -> Iterator[str]:
    """Serve the real Workbench assets and API on an ephemeral local port."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    app = build_workbench_api_app(ocscore_root, max_depth=3, server_port=port)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        name="workbench-browser-test-server",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        raise RuntimeError("Workbench browser test server did not start")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        if thread.is_alive():
            pytest.fail("Workbench browser test server did not stop")


# Functions
###############################################################################
## Private ##


def _prepare_page(page: object) -> tuple[list[str], list[str]]:
    """Stub Plotly and collect uncaught browser errors."""

    console_errors: list[str] = []
    page_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text) if message.type == "error" else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.route(
        "https://cdn.plot.ly/**",
        lambda route: route.fulfill(
            status=200,
            content_type="application/javascript",
            body=PLOTLY_STUB,
        ),
    )
    return console_errors, page_errors


def _open_loaded_workbench(page: object, workbench_url: str) -> None:
    """Open the dashboard and wait until its initial refresh is complete."""

    page.goto(f"{workbench_url}/app")
    page.locator("#health-label").wait_for(state="visible")
    playwright.expect(page.locator("#health-label")).to_have_text("Strict OCScore layout")
    playwright.expect(page.locator("#study-count")).to_have_text("2")


# Tests
###############################################################################


@pytest.mark.browser
def test_workbench_bootstrap_renders_real_workspace_without_browser_errors(
    page: object,
    workbench_url: str,
    ocscore_root: Path,
) -> None:
    """The DOM bootstrap fetches and renders the strict workspace payload."""

    console_errors, page_errors = _prepare_page(page)
    _open_loaded_workbench(page, workbench_url)

    playwright.expect(page).to_have_title("OCDocker Workbench")
    playwright.expect(page.get_by_role("heading", name="OCScore Control Dashboard")).to_be_visible()
    playwright.expect(page.locator("#health-dot")).to_have_class("dot ok")
    playwright.expect(page.locator("#root-label")).to_have_text(ocscore_root.name)
    playwright.expect(page.locator("#root-label")).to_have_attribute("title", str(ocscore_root))
    playwright.expect(page.locator("#completed-count")).to_have_text("1")
    playwright.expect(page.locator("#failed-count")).to_have_text("0")
    playwright.expect(page.locator("#missing-count")).to_have_text("0")
    playwright.expect(page.locator("#comparison-summary")).to_contain_text("2 models")
    playwright.expect(page.locator("#comparison-summary")).to_contain_text("vs full_ocscore")
    playwright.expect(page.locator("#comparison-table")).to_contain_text("full_ocscore")
    playwright.expect(page.locator("#comparison-table")).to_contain_text("no_ligand")
    playwright.expect(page.locator(".plotly-characterization-stub").first).to_be_visible()
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_workbench_persists_theme_tab_and_collapsed_zone_across_reload(
    page: object,
    workbench_url: str,
) -> None:
    """User-facing navigation state survives a full document reload."""

    console_errors, page_errors = _prepare_page(page)
    _open_loaded_workbench(page, workbench_url)

    page.locator("#theme-toggle").click()
    playwright.expect(page.locator("html")).to_have_attribute("data-theme", "light")
    protocol = page.locator('[data-zone="protocol"]')
    protocol.locator(".zone-toggle").click()
    playwright.expect(protocol).to_have_class(re.compile(r"\bis-collapsed\b"))
    page.locator("#tab-design").click()
    playwright.expect(page.locator("#tab-design")).to_have_attribute("aria-selected", "true")
    playwright.expect(page.locator("#panel-design")).to_be_visible()

    saved = json.loads(page.evaluate("localStorage.getItem('ocscore-workbench-ui')"))
    assert saved["theme"] == "light"
    assert saved["activeTab"] == "design"
    assert saved["zoneCollapsed"]["protocol"] is True

    page.reload()
    playwright.expect(page.locator("#health-label")).to_have_text("Strict OCScore layout")
    playwright.expect(page.locator("html")).to_have_attribute("data-theme", "light")
    playwright.expect(page.locator("#tab-design")).to_have_attribute("aria-selected", "true")
    playwright.expect(page.locator("#panel-design")).to_be_visible()
    playwright.expect(protocol).to_have_class(re.compile(r"\bis-collapsed\b"))
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_workbench_comparison_selection_and_sort_survive_reload(
    page: object,
    workbench_url: str,
) -> None:
    """Comparison-table interactions keep the selected study and sort order."""

    console_errors, page_errors = _prepare_page(page)
    _open_loaded_workbench(page, workbench_url)

    model_sort = '#comparison-table button[data-sort-key="model"]'
    page.locator(model_sort).click()
    page.locator(model_sort).click()
    playwright.expect(page.locator(model_sort)).to_contain_text("desc")

    ablation_row = page.locator('#comparison-table tbody tr[data-entry-id="no_ligand"]')
    ablation_row.click()
    playwright.expect(page.locator("#detail-panel")).to_be_visible()
    playwright.expect(page.locator("#detail-title")).to_have_text("no_ligand — replicas & figures")
    playwright.expect(page.locator("#run-context-items")).to_contain_text("no_ligand")

    saved = json.loads(page.evaluate("localStorage.getItem('ocscore-workbench-ui')"))
    assert saved["comparisonSort"] == {"key": "model", "direction": "desc"}
    assert saved["selectedStudyName"] == "no_ligand"

    page.reload()
    playwright.expect(page.locator("#health-label")).to_have_text("Strict OCScore layout")
    playwright.expect(page.locator(model_sort)).to_contain_text("desc")
    playwright.expect(page.locator("#detail-title")).to_have_text("no_ligand — replicas & figures")
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_workbench_jobs_tab_persists_and_clears_local_token(
    page: object,
    workbench_url: str,
) -> None:
    """The jobs panel binds its API refresh and local token controls."""

    console_errors, page_errors = _prepare_page(page)
    _open_loaded_workbench(page, workbench_url)

    page.locator("#tab-jobs").click()
    playwright.expect(page.locator("#panel-jobs")).to_be_visible()
    playwright.expect(page.locator("#jobs-summary")).to_have_text("0 jobs · 0 running")
    page.locator("#jobs-token-input").fill("characterization-token")
    page.locator("#jobs-token-save").click()
    playwright.expect(page.locator("#jobs-token-status")).to_have_text("Token configured")

    saved = json.loads(page.evaluate("localStorage.getItem('ocscore-workbench-ui')"))
    assert saved["activeTab"] == "jobs"
    assert saved["jobToken"] == "characterization-token"

    page.reload()
    playwright.expect(page.locator("#health-label")).to_have_text("Strict OCScore layout")
    playwright.expect(page.locator("#panel-jobs")).to_be_visible()
    playwright.expect(page.locator("#jobs-token-input")).to_have_value("characterization-token")
    page.locator("#jobs-token-clear").click()
    playwright.expect(page.locator("#jobs-token-status")).to_contain_text("No token set")
    playwright.expect(page.locator("#jobs-token-input")).to_have_value("")
    cleared = json.loads(page.evaluate("localStorage.getItem('ocscore-workbench-ui')"))
    assert cleared["jobToken"] == ""
    assert page_errors == []
    assert console_errors == []


@pytest.mark.browser
def test_workbench_surfaces_workspace_refresh_failure(page: object, workbench_url: str) -> None:
    """A failed workspace request updates health and exposes the API error."""

    console_errors, page_errors = _prepare_page(page)
    page.route(
        "**/api/ocscore-workspace",
        lambda route: route.fulfill(
            status=500,
            content_type="application/json",
            body=json.dumps({"ok": False, "error": "Synthetic workspace failure"}),
        ),
    )
    page.goto(f"{workbench_url}/app")

    playwright.expect(page.locator("#health-label")).to_have_text("Error")
    playwright.expect(page.locator("#health-dot")).to_have_class("dot error")
    playwright.expect(page.locator("#toast")).to_have_text("Synthetic workspace failure")
    playwright.expect(page.locator("#toast")).to_have_class(re.compile(r"\bshow\b"))
    assert page_errors == []
    assert len(console_errors) == 1
    assert "500 (Internal Server Error)" in console_errors[0]
