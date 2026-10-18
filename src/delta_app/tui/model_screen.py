"""Modal picker for switching provider and model.

A three-step dialog — provider, then model, then an API key if one is not
already stored. The model step lists the catalog's models (suggested ones
first) with a filter box; Enter on a name that is not listed uses it as
typed. A catalog refresh runs in the background while the dialog is open.
It gathers a selection and dismisses with the result; the app performs the
actual switch through the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView

from delta_app.catalog import CatalogError, available_providers, models_for, refresh_catalog
from delta_app.config.store import ProviderInfo, load_config


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """The result of a completed picker run."""

    provider: str
    label: str
    model: str
    api_key: str | None


class _Row(ListItem):
    """A list row carrying an opaque value."""

    def __init__(self, text: str, value: str) -> None:
        super().__init__(Label(text))
        self.value = value


class ModelScreen(ModalScreen[ModelChoice | None]):
    """Small centred dialog: pick a provider, a model, and a key if needed."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self, current_provider: str = "", current_model: str = "", *, login: bool = False,
    ) -> None:
        super().__init__()
        self._current_provider = current_provider
        self._current_model = current_model
        self._info: ProviderInfo | None = None
        self._model: str = ""
        self._login = login
        self._providers = available_providers()

    # -- composition --------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            yield Label("Select provider", id="model-title")
            yield Input(placeholder="Filter models, or type a model id", id="model-filter")
            yield ListView(id="model-list")
            yield Input(placeholder="API key", password=True, id="model-key")
            yield Label("", id="model-hint")

    def on_mount(self) -> None:
        self.query_one("#model-key", Input).display = False
        self.query_one("#model-filter", Input).display = False
        self._show_providers()
        # Never blocks the dialog; a failed or skipped refresh changes nothing.
        self.run_worker(self._refresh_catalog(), exclusive=True)

    async def _refresh_catalog(self) -> None:
        try:
            result = await refresh_catalog()
        except CatalogError:
            return
        if result.refreshed and self._info is not None and self.is_attached:
            self._fill_models(self.query_one("#model-filter", Input).value)

    # -- steps --------------------------------------------------------------

    def _show_providers(self) -> None:
        """Step 1 — list the supported providers."""
        self.query_one("#model-title", Label).update("Select provider")
        listing = self.query_one("#model-list", ListView)
        listing.clear()
        for info in self._providers:
            marker = "●" if info.key == self._current_provider else " "
            listing.append(_Row(f"{marker} {info.label}", info.key))
        listing.index = 0
        listing.focus()

    def _show_models(self, info: ProviderInfo) -> None:
        """Step 2 — list that provider's models, with a filter box."""
        self._info = info
        self.query_one("#model-title", Label).update(f"{info.label} — select model")
        self.query_one("#model-hint", Label).update(
            "Type to filter · Enter picks · a name not listed is used as typed"
        )
        field = self.query_one("#model-filter", Input)
        field.display = True
        field.value = ""
        self._fill_models("")
        field.focus()

    def _fill_models(self, needle: str) -> None:
        assert self._info is not None
        wanted = needle.strip().lower()
        listing = self.query_one("#model-list", ListView)
        listing.clear()
        for name in models_for(self._info.key):
            if wanted and wanted not in name.lower():
                continue
            marker = "●" if name == self._current_model else " "
            listing.append(_Row(f"{marker} {name}", name))
        listing.index = 0

    def _show_key_prompt(self) -> None:
        """Step 3 — ask for an API key, only when none is stored."""
        assert self._info is not None
        self.query_one("#model-list", ListView).display = False
        self.query_one("#model-filter", Input).display = False
        field = self.query_one("#model-key", Input)
        field.display = True
        field.focus()
        self.query_one("#model-title", Label).update(f"{self._info.label} — API key")
        self.query_one("#model-hint", Label).update("Enter to save · Esc to cancel")

    def _stored_key(self, info: ProviderInfo) -> str | None:
        """Key already saved in config or the environment, if any."""
        import os

        return os.environ.get(info.key_env) or load_config().api_keys.get(info.key)

    # -- events -------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        row = event.item
        if not isinstance(row, _Row):
            return
        if self._info is None:
            self._on_provider_chosen(row.value)
        else:
            self._on_model_chosen(row.value)

    def _on_provider_chosen(self, key: str) -> None:
        info = next((p for p in self._providers if p.key == key), None)
        if info is None:
            return
        self._show_models(info)

    def _on_model_chosen(self, model: str) -> None:
        assert self._info is not None
        self._model = model
        if not self._info.needs_key or (not self._login and self._stored_key(self._info)):
            self._finish(None)
            return
        self._show_key_prompt()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "model-filter" and self._info is not None:
            event.stop()
            self._fill_models(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if event.input.id == "model-filter":
            self._on_filter_submitted(event.value)
            return
        key = event.value.strip()
        if not key:
            self.query_one("#model-hint", Label).update("API key must not be empty.")
            return
        self._finish(key)

    def _on_filter_submitted(self, text: str) -> None:
        """Enter in the filter: the highlighted match, else the typed id."""
        listing = self.query_one("#model-list", ListView)
        row = listing.highlighted_child
        if isinstance(row, _Row):
            self._on_model_chosen(row.value)
        elif text.strip():
            self._on_model_chosen(text.strip())

    def _finish(self, api_key: str | None) -> None:
        assert self._info is not None
        self.dismiss(
            ModelChoice(
                provider=self._info.key,
                label=self._info.label,
                model=self._model,
                api_key=api_key,
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["ModelChoice", "ModelScreen"]
