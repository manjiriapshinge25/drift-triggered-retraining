# AWS Lambda deployment checklist (requires your own AWS account)

This is an implementation guide, not proof that a cloud deployment has already occurred.

1. Install AWS CLI and AWS SAM CLI, configure credentials with a least-privilege IAM identity, and select a region.
2. Build in an Amazon Linux-compatible environment if native package wheels are needed. Include `endpoint/`, `src/`, the `artifacts/registry.json` file, and all referenced `artifacts/models/*.joblib` files in the package. Include runtime dependencies (`pandas`, `numpy`, `scikit-learn`, `joblib`) compatible with Python 3.12 and the Lambda architecture.
3. Review `template.yaml`; it creates a Lambda function and HTTP API routes for `/predict` and `/health`. Do not commit credentials or `.aws` files.
4. From the project root, run `sam build` and then `sam deploy --guided`. Review all resource and cost prompts before approving.
5. Copy the API URL from the stack outputs. Call `GET /health`, then `POST /predict` with a valid `features` JSON object. Record the returned model version.
6. For true provider measurements, collect at least 30 cold invocations (separate fresh execution environments; Lambda does not guarantee each invocation is cold) and 30 warm invocations. Record request IDs, duration, init duration when exposed, model version, and timestamps. Do not label local notebook latency as cloud latency.
7. Promote a candidate only through the workflow and registry update. Repeat `/health` and `/predict` to show the version changed while the handler stayed unchanged.
8. Save the deployment stack output and redacted invocation logs as submission evidence. Never publish account IDs, tokens, or credentials.

**Important packaging note:** a scikit-learn/pandas Lambda package can exceed the direct ZIP limit. Use a Lambda container image or Lambda layers if required. This needs to be tested in your account and is not completed by running the local notebook.
