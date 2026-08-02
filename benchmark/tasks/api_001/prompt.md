We're adding address validation via an external provider. Their OpenAPI specification is checked in at `docs/vendor/address-api.yaml`. There's no sandbox account yet, so you can't call the real service.

Write the client integration.

Requirements:

- Cover the endpoints described in the spec that we need for validation — read the spec to determine which those are.
- Authenticate the way the spec requires; the credential must come from configuration, never be hardcoded.
- Map the provider's error responses onto our own exception types rather than leaking theirs upward.
- Tests must not require network access.
