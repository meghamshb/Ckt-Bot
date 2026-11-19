# Agent setup

When the user says **setup**, install and start K-Bot by following [run.md](run.md). Run the commands yourself using your terminal tools. Ask the user only for sign-in, required permissions, or a blocking error you cannot resolve.

- Check Apple Silicon, macOS 14 or later, disk space and the required system tools first.
- Use the pinned package versions, source commits, model revision and CLI version in `run.md`.
- Keep models, environments, credentials and outputs outside the Git checkout. Never copy paths or credentials from the original developer's machine.
- Reuse an existing installation only after its checks pass. Do not overwrite or delete an existing runtime, model, dataset or environment. Use a new output folder when a fresh configuration is needed.
- Check the official CLI login status. If sign-in is needed, start device login and give the user its current URL and code. Never request a token in chat.
- Run `pip check`, configure the private runtime, run `check-ready`, then start the web server on an available local port. Preserve the configured `CODEX_HOME` when starting the server.
- If asked to verify the installation, upload the example from `run.md` through the real interface and check the result. Do not substitute saved answers or report readiness as an end-to-end pass.
- Finish with the website URL, the setup result and any unresolved blocker.

Keep documentation short and use plain language without emojis or em dashes. Preserve the working pipeline, model checks and third-party license notices.
