# Two Tower Adapter Compatibility

This adapter depends on the exported Two Tower artifacts contract produced by training:

- `user_tower_state.pt`
- `vocab_artifact.pkl`
- `client_embeddings.parquet`

## Supported artifact layouts

The adapter expects the artifact URIs created by:

- `two_tower.inference.artifact_paths.training_artifact_uris(artifacts_base)`

Which resolves to:

- `{artifacts_base}/artifacts/user_tower/user_tower_state.pt`
- `{artifacts_base}/artifacts/vocab_artifact/vocab_artifact.pkl`
- `{artifacts_base}/artifacts/client_embeddings/client_embeddings.parquet`

## Versioning guidance

- Analytics contract versions:
  - `prediction_schema_version`
  - `feature_catalog_version`
- Adapter version should be bumped when the Two Tower artifact layout changes.

## Known limitations

- True rescoring uses **user tower + fixed client embeddings**.
- Client-side feature importance is not computed unless the client tower checkpoint is also exported.

