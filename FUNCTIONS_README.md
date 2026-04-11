# ChaosEngine Function Reference

Auto-generated from the current repository source tree.

## Scope

- Inventories first-party functions, classes, methods, components, scripts, and Rust handlers.
- Covers backend_service, src, src-tauri/src, scripts, tests, and demo scripts.
- Excludes vendored/build output such as node_modules, dist, .venv, and .git.

## Totals

- Source files indexed: 22
- Python files: 6
- TypeScript/TSX/MJS files: 14
- Rust files: 2

## Python modules

### `backend_service/__init__.py`

> ChaosEngineAI desktop backend service.

- Top-level functions: 0
- Classes: 0

### `backend_service/app.py`

- Top-level functions: 43
- Classes: 13

#### Functions

- `_flatten_catalog() -> list[dict[str, Any]]`
- `_resolve_app_version() -> str`
- `_normalize_slug(value: str, fallback: str) -> str`
- `_default_settings() -> dict[str, Any]`
- `_normalize_model_directory_entry(entry: dict[str, Any], index: int) -> dict[str, Any]`
- `_normalize_model_directories(entries: list[dict[str, Any]]) -> list[dict[str, Any]]`
- `_normalize_launch_preferences(payload: dict[str, Any] | None) -> dict[str, Any]`
- `_load_settings(path: Path=SETTINGS_PATH) -> dict[str, Any]`
- `_save_settings(settings: dict[str, Any], path: Path=SETTINGS_PATH) -> None`
- `_bytes_to_gb(value: int | float) -> float`
- `_safe_run(command: list[str], timeout: float=1.5) -> str | None`
- `_apple_hardware_summary(total_memory_gb: float) -> str | None`
- `_generic_hardware_summary(total_memory_gb: float) -> str`
- `_hf_repo_from_link(link: str | None) -> str | None`
- `_runtime_label(capabilities: dict[str, Any] | None=None) -> str`
- `_detect_gpu_utilization() -> float | None`
- `_list_llm_processes(limit: int=12) -> list[dict[str, Any]]`
- `_build_system_snapshot() -> dict[str, Any]`
- `_best_fit_recommendation(system_stats: dict[str, Any]) -> dict[str, Any]`
- `_path_size_bytes(path: Path, *, seen: set[tuple[int, int]] | None=None) -> int`
- `_du_size_gb(path: Path) -> float`
- `_relative_depth(path: Path, root: Path) -> int`
- `_detect_directory_model(path: Path) -> tuple[str, str] | None`
- `_iter_discovered_models(root: Path, *, max_depth: int=8) -> list[tuple[Path, str, str]]`
- `_discover_local_models(model_directories: list[dict[str, Any]], limit: int=500) -> list[dict[str, Any]]`
- `_reveal_path_in_file_manager(path: Path) -> None`
- `_estimate_runtime_memory_gb(params_b: float, quantization: str) -> float`
- `_variant_available_locally(variant: dict[str, Any], library: list[dict[str, Any]]) -> bool`
- `_model_family_payloads(system_stats: dict[str, Any], library: list[dict[str, Any]]) -> list[dict[str, Any]]`
- `_search_huggingface_hub(query: str, library: list[dict[str, Any]], limit: int=20) -> list[dict[str, Any]] — Search HuggingFace Hub for models matching the query.`
- `_default_chat_variant() -> dict[str, Any]`
- `_seed_chat_sessions() -> list[dict[str, Any]]`
- `_context_label(value: int | None) -> str`
- `_benchmark_label(model_name: str, *, cache_strategy: str, bits: int, fp16_layers: int, context_tokens: int) -> str`
- `_seed_benchmark_runs() -> list[dict[str, Any]]`
- `_load_benchmark_runs(path: Path=BENCHMARKS_PATH) -> list[dict[str, Any]]`
- `_save_benchmark_runs(runs: list[dict[str, Any]], path: Path=BENCHMARKS_PATH) -> None`
- `_local_ipv4_addresses() -> list[str]`
- `_estimate_baseline_tok_s(system_stats: dict[str, Any]) -> float`
- `compute_cache_preview(*, bits: int=3, fp16_layers: int=4, num_layers: int=32, num_heads: int=32, hidden_size: int=4096, context_tokens: int=8192, params_b: float=7.0, system_stats: dict[str, Any] | None=None) -> dict[str, Any]`
- `_build_benchmarks() -> list[dict[str, Any]]`
- `create_app(state: ChaosEngineState | None=None) -> FastAPI`
- `main() -> None`

#### Classes and methods

- `LoadModelRequest` (BaseModel)
- `ModelDirectoryRequest` (BaseModel)
- `LaunchPreferencesRequest` (BaseModel)
- `CreateSessionRequest` (BaseModel)
- `UpdateSessionRequest` (BaseModel)
- `GenerateRequest` (BaseModel)
- `UpdateSettingsRequest` (BaseModel)
- `OpenAIMessage` (BaseModel)
- `OpenAIChatCompletionRequest` (BaseModel)
- `ConvertModelRequest` (BaseModel)
- `BenchmarkRunRequest` (BaseModel)
- `RevealPathRequest` (BaseModel)
- `ChaosEngineState`
  - `__init__(self, *, system_snapshot_provider=_build_system_snapshot, library_provider=None, server_port: int=DEFAULT_PORT, settings_path: Path=SETTINGS_PATH, benchmarks_path: Path=BENCHMARKS_PATH) -> None`
  - `_launch_preferences(self) -> dict[str, Any]`
  - `_library(self, *, force: bool=False) -> list[dict[str, Any]]`
  - `_settings_payload(self, library: list[dict[str, Any]]) -> dict[str, Any]`
  - `_bootstrap(self) -> None`
  - `_time_label() -> str`
  - `_relative_label() -> str`
  - `add_log(self, source: str, level: str, message: str) -> None`
  - `subscribe_logs(self)`
  - `unsubscribe_logs(self, q) -> None`
  - `add_activity(self, title: str, detail: str) -> None`
  - `_cache_strategy_label(self, bits: int, fp16_layers: int) -> str`
  - `_native_cache_label() -> str`
  - `_cache_label(self, *, cache_strategy: str, bits: int, fp16_layers: int) -> str`
  - `_assistant_metrics_payload(self, result: Any) -> dict[str, Any]`
  - `_should_reload_for_profile(self, *, model_ref: str | None, cache_bits: int, fp16_layers: int, fused_attention: bool, cache_strategy: str, fit_model_in_memory: bool, context_tokens: int) -> bool`
  - `_append_benchmark_run(self, run: dict[str, Any]) -> None`
  - `_find_catalog_entry(self, model_ref: str) -> dict[str, Any] | None`
  - `_find_library_entry(self, path: str | None, model_ref: str | None) -> dict[str, Any] | None`
  - `_resolve_model_target(self, *, model_ref: str | None, path: str | None, backend: str) -> tuple[str | None, str]`
  - `_default_session_model(self) -> dict[str, Any]`
  - `_promote_session(self, session: dict[str, Any]) -> None`
  - `_ensure_session(self, session_id: str | None=None, title: str | None=None) -> dict[str, Any]`
  - `create_session(self, title: str | None=None) -> dict[str, Any]`
  - `update_session(self, session_id: str, request: UpdateSessionRequest) -> dict[str, Any]`
  - `update_settings(self, request: UpdateSettingsRequest) -> dict[str, Any]`
  - `_conversion_details(self, *, request: ConvertModelRequest, conversion: dict[str, Any]) -> dict[str, Any]`
  - `run_benchmark(self, request: BenchmarkRunRequest) -> dict[str, Any]`
  - `load_model(self, request: LoadModelRequest) -> dict[str, Any]`
  - `unload_model(self) -> dict[str, Any]`
  - `convert_model(self, request: ConvertModelRequest) -> dict[str, Any]`
  - `reveal_model_path(self, path: str) -> dict[str, Any]`
  - `generate(self, request: GenerateRequest) -> dict[str, Any]`
  - `server_status(self) -> dict[str, Any]`
  - `workspace(self) -> dict[str, Any]`
  - `openai_models(self) -> dict[str, Any]`
  - `openai_chat_completion(self, request: OpenAIChatCompletionRequest) -> dict[str, Any] | StreamingResponse`

### `backend_service/inference.py`

- Top-level functions: 14
- Classes: 10

#### Functions

- `_now_label() -> str`
- `_normalize_message_content(content: Any) -> str`
- `_read_text_tail(path: Path | None, limit: int=40) -> str`
- `_json_subprocess(command: list[str], *, timeout: float=15.0, cwd: Path=WORKSPACE_ROOT) -> tuple[int, dict[str, Any] | None, str]`
- `_find_open_port() -> int`
- `_resolve_mlx_python() -> str`
- `_resolve_llama_server() -> str | None`
- `_resolve_llama_cli() -> str | None`
- `_http_json(url: str, *, payload: dict[str, Any] | None=None, timeout: float=30.0) -> dict[str, Any]`
- `_looks_like_gguf(value: str | None) -> bool`
- `_default_conversion_output(source_label: str) -> str`
- `_path_size_bytes(path: str | Path | None) -> int`
- `_probe_native_backends() -> BackendCapabilities`
- `get_backend_capabilities(*, force: bool=False) -> BackendCapabilities`

#### Classes and methods

- `BackendCapabilities`
  - `to_dict(self) -> dict[str, Any]`
- `LoadedModelInfo`
  - `to_dict(self) -> dict[str, Any]`
- `GenerationResult`
  - `to_metrics(self) -> dict[str, Any]`
- `StreamChunk`
- `BaseInferenceEngine`
  - `load_model(self, *, model_ref: str, model_name: str, source: str, backend: str, path: str | None, runtime_target: str | None, cache_bits: int, fp16_layers: int, fused_attention: bool, cache_strategy: str, fit_model_in_memory: bool, context_tokens: int) -> LoadedModelInfo`
  - `unload_model(self) -> None`
  - `generate(self, *, prompt: str, history: list[dict[str, Any]], system_prompt: str | None, max_tokens: int, temperature: float) -> GenerationResult`
  - `stream_generate(self, *, prompt: str, history: list[dict[str, Any]], system_prompt: str | None, max_tokens: int, temperature: float) -> Iterator[StreamChunk]`
- `MockInferenceEngine` (BaseInferenceEngine)
  - `__init__(self, capabilities: BackendCapabilities) -> None`
  - `load_model(self, *, model_ref: str, model_name: str, source: str, backend: str, path: str | None, runtime_target: str | None, cache_bits: int, fp16_layers: int, fused_attention: bool, cache_strategy: str, fit_model_in_memory: bool, context_tokens: int) -> LoadedModelInfo`
  - `unload_model(self) -> None`
  - `generate(self, *, prompt: str, history: list[dict[str, Any]], system_prompt: str | None, max_tokens: int, temperature: float) -> GenerationResult`
- `JsonRpcProcess`
  - `__init__(self, command: list[str], *, timeout: float=DEFAULT_MLX_TIMEOUT_SECONDS) -> None`
  - `_pump_stdout(self) -> None`
  - `start(self) -> None`
  - `close(self) -> None`
  - `request(self, payload: dict[str, Any]) -> dict[str, Any]`
  - `stream_request(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]`
- `MLXWorkerEngine` (BaseInferenceEngine)
  - `__init__(self, capabilities: BackendCapabilities) -> None`
  - `load_model(self, *, model_ref: str, model_name: str, source: str, backend: str, path: str | None, runtime_target: str | None, cache_bits: int, fp16_layers: int, fused_attention: bool, cache_strategy: str, fit_model_in_memory: bool, context_tokens: int) -> LoadedModelInfo`
  - `unload_model(self) -> None`
  - `generate(self, *, prompt: str, history: list[dict[str, Any]], system_prompt: str | None, max_tokens: int, temperature: float) -> GenerationResult`
  - `stream_generate(self, *, prompt: str, history: list[dict[str, Any]], system_prompt: str | None, max_tokens: int, temperature: float) -> Iterator[StreamChunk]`
- `LlamaCppEngine` (BaseInferenceEngine)
  - `__init__(self, capabilities: BackendCapabilities) -> None`
  - `_server_url(self, path: str) -> str`
  - `_cleanup_process(self) -> None`
  - `_build_command(self, *, path: str | None, runtime_target: str | None, cache_bits: int, context_tokens: int, use_cache_compression: bool, fit_enabled: bool, retry_without_cache: bool) -> tuple[list[str], int, str | None]`
  - `_wait_for_server(self) -> None`
  - `load_model(self, *, model_ref: str, model_name: str, source: str, backend: str, path: str | None, runtime_target: str | None, cache_bits: int, fp16_layers: int, fused_attention: bool, cache_strategy: str, fit_model_in_memory: bool, context_tokens: int) -> LoadedModelInfo`
  - `unload_model(self) -> None`
  - `generate(self, *, prompt: str, history: list[dict[str, Any]], system_prompt: str | None, max_tokens: int, temperature: float) -> GenerationResult`
  - `stream_generate(self, *, prompt: str, history: list[dict[str, Any]], system_prompt: str | None, max_tokens: int, temperature: float) -> Iterator[StreamChunk]`
- `RuntimeController`
  - `__init__(self) -> None`
  - `refresh_capabilities(self, *, force: bool=False) -> BackendCapabilities`
  - `_make_mock_engine(self) -> MockInferenceEngine`
  - `_select_engine(self, *, backend: str, runtime_target: str | None, path: str | None) -> BaseInferenceEngine`
  - `_display_name(model_ref: str, model_name: str | None=None, path: str | None=None) -> str`
  - `_is_same_loaded_model(self, model_ref: str | None) -> bool`
  - `load_model(self, *, model_ref: str, model_name: str | None=None, source: str='catalog', backend: str='auto', path: str | None=None, runtime_target: str | None=None, cache_bits: int=3, fp16_layers: int=4, fused_attention: bool=False, cache_strategy: str='auto', fit_model_in_memory: bool=True, context_tokens: int=8192) -> LoadedModelInfo`
  - `unload_model(self) -> None`
  - `generate(self, *, prompt: str, history: list[dict[str, Any]], system_prompt: str | None, max_tokens: int, temperature: float) -> GenerationResult`
  - `stream_generate(self, *, prompt: str, history: list[dict[str, Any]], system_prompt: str | None, max_tokens: int, temperature: float) -> Iterator[StreamChunk]`
  - `extract_gguf_metadata(self, path: str) -> dict[str, Any]`
  - `convert_model(self, *, source_ref: str | None, source_path: str | None, output_path: str | None, hf_repo: str | None, quantize: bool, q_bits: int, dtype: str) -> dict[str, Any]`
  - `status(self, *, active_requests: int=0, requests_served: int=0) -> dict[str, Any]`

### `backend_service/mlx_worker.py`

- Top-level functions: 7
- Classes: 1

#### Functions

- `_normalize_message_content(content: Any) -> str`
- `_build_prompt_text(tokenizer: Any, history: list[dict[str, Any]], prompt: str, system_prompt: str | None) -> str`
- `_emit(payload: dict[str, Any]) -> None`
- `probe() -> int`
- `gguf_metadata(path: str) -> int`
- `serve() -> int`
- `main(argv: list[str] | None=None) -> int`

#### Classes and methods

- `WorkerState`
  - `__init__(self) -> None`
  - `handle(self, request: dict[str, Any]) -> dict[str, Any] | None`
  - `load_model(self, request: dict[str, Any]) -> dict[str, Any]`
  - `unload_model(self) -> dict[str, Any]`
  - `generate(self, request: dict[str, Any]) -> dict[str, Any]`
  - `stream_generate(self, request: dict[str, Any]) -> None`

### `tests/test_backend_service.py`

- Top-level functions: 2
- Classes: 1

#### Functions

- `fake_system_snapshot()`
- `fake_library()`

#### Classes and methods

- `ChaosEngineBackendTests` (unittest.TestCase)
  - `setUp(self)`
  - `tearDown(self)`
  - `test_health_reports_runtime_metadata(self)`
  - `test_model_load_and_chat_generation(self)`
  - `test_openai_compatible_completion_autoloads_model(self)`
  - `test_preview_math_reduces_cache_size(self)`
  - `test_convert_endpoint_returns_conversion_payload(self)`
  - `test_session_update_renames_and_switches_model(self)`
  - `test_settings_endpoint_updates_model_directories_and_launch_defaults(self)`
  - `test_explicit_gguf_path_wins_over_hf_cache_library_entry(self)`
  - `test_recursive_discovery_finds_nested_model_directories(self)`
  - `test_reveal_model_path_endpoint_returns_resolved_path(self)`

### `tests/test_speed.py`

> Speed comparison: Python vs Metal dequant in cache context.

- Top-level functions: 3
- Classes: 0

#### Functions

- `bench_dequant(seq_len, dim=128, bits=3, n_iters=50) — Compare Python vs Metal dequant at given sequence length.`
- `bench_full_cache_cycle(seq_len, n_heads=32, dim=128, bits=3, n_iters=20) — Simulate decode: quantize 1 token + dequant full cache.`
- `main()`

## Frontend and build scripts

### `scripts/load-env.mjs`

- Exported symbols found: 1
- Function-like declarations found: 2
- Classes found: 0

#### Exports

- Line 5: `function loadEnvFiles`

#### Function-like declarations

- Line 14: `loadEnvFile` — `function loadEnvFile(filePath) {`
- Line 37: `normalizeValue` — `function normalizeValue(rawValue) {`

### `scripts/release-macos.mjs`

- Exported symbols found: 0
- Function-like declarations found: 11
- Classes found: 0

#### Function-like declarations

- Line 37: `main` — `function main() {`
- Line 81: `normalizeArch` — `function normalizeArch(value) {`
- Line 91: `resolveNotaryAuthArgs` — `function resolveNotaryAuthArgs() {`
- Line 126: `signAppBundle` — `function signAppBundle(appPath) {`
- Line 143: `signFlatArtifact` — `function signFlatArtifact(targetPath) {`
- Line 147: `verifySignedApp` — `function verifySignedApp(appPath) {`
- Line 152: `createDistributionDmg` — `function createDistributionDmg(appPath, dmgPath) {`
- Line 176: `notarizeArtifact` — `function notarizeArtifact(targetPath, authArgs) {`
- Line 196: `run` — `function run(command, commandArgs, options = {}) {`
- Line 204: `ensureDir` — `function ensureDir(targetPath) {`
- Line 208: `assertPathExists` — `function assertPathExists(targetPath, label) {`

### `scripts/stage-runtime.mjs`

- Exported symbols found: 0
- Function-like declarations found: 25
- Classes found: 0

#### Function-like declarations

- Line 39: `main` — `function main() {`
- Line 111: `resolvePlatformTag` — `function resolvePlatformTag() {`
- Line 123: `normalizeArch` — `function normalizeArch(arch) {`
- Line 133: `inspectPython` — `function inspectPython() {`
- Line 159: `stageLlamaBinaries` — `function stageLlamaBinaries() {`
- Line 189: `defaultLlamaBinDir` — `function defaultLlamaBinDir() {`
- Line 193: `shouldCopyLlamaEntry` — `function shouldCopyLlamaEntry(entry) {`
- Line 209: `binaryName` — `function binaryName(base) {`
- Line 213: `resolveEmbeddedPythonBinary` — `function resolveEmbeddedPythonBinary(versionTag) {`
- Line 230: `resolveExistingPath` — `function resolveExistingPath(targetPath, label) {`
- Line 237: `assertPathExists` — `function assertPathExists(targetPath, label) {`
- Line 243: `cleanupStaleTauriResources` — `function cleanupStaleTauriResources() {`
- Line 257: `cleanupStagedRuntimeArtifacts` — `function cleanupStagedRuntimeArtifacts() {`
- Line 270: `maybeSignEmbeddedRuntime` — `function maybeSignEmbeddedRuntime() {`
- Line 295: `collectSignTargets` — `function collectSignTargets(rootPath) {`
- Line 320: `walk` — `function walk(currentPath, visitor) {`
- Line 332: `looksLikeBundle` — `function looksLikeBundle(targetPath) {`
- Line 336: `isRuntimeBinary` — `function isRuntimeBinary(targetPath, stat) {`
- Line 349: `pathDepth` — `function pathDepth(targetPath) {`
- Line 353: `copyTree` — `function copyTree(source, destination) {`
- Line 363: `copyFile` — `function copyFile(source, destination) {`
- Line 368: `copyPath` — `function copyPath(source, destination) {`
- Line 384: `safeUnlink` — `function safeUnlink(targetPath) {`
- Line 392: `shouldIgnorePath` — `function shouldIgnorePath(currentPath) {`
- Line 408: `ensureDir` — `function ensureDir(targetPath) {`

### `src/App.tsx`

- Exported symbols found: 1
- Function-like declarations found: 83
- Classes found: 0

#### Exports

- Line 315: `default function App`

#### Function-like declarations

- Line 71: `number` — `function number(value: number, digits = 1) {`
- Line 75: `serverOriginFromBase` — `function serverOriginFromBase(baseUrl: string) {`
- Line 79: `serverBaseFromOrigin` — `function serverBaseFromOrigin(origin: string) {`
- Line 83: `upsertSession` — `function upsertSession(sessions: ChatSession[], nextSession: ChatSession): ChatSession[] {`
- Line 87: `sessionPreview` — `function sessionPreview(session: ChatSession) {`
- Line 91: `sortSessions` — `function sortSessions(sessions: ChatSession[]) {`
- Line 100: `syncRuntime` — `function syncRuntime(current: WorkspaceData, runtime: RuntimeStatus): WorkspaceData {`
- Line 116: `syncStoppedBackend` — `function syncStoppedBackend(current: WorkspaceData, runtimeInfo: TauriBackendInfo | null): WorkspaceData {`
- Line 149: `flattenVariants` — `function flattenVariants(families: ModelFamily[]): ModelVariant[] {`
- Line 153: `normalizeCapability` — `function normalizeCapability(capability: string): string {`
- Line 157: `capabilityMeta` — `function capabilityMeta(capability: string) {`
- Line 167: `sizeLabel` — `function sizeLabel(sizeGb: number) {`
- Line 171: `tokenSet` — `function tokenSet(value: string): string[] {`
- Line 178: `libraryItemMatchesVariant` — `function libraryItemMatchesVariant(item: LibraryItem, variant: ModelVariant): boolean {`
- Line 198: `findLibraryItemForVariant` — `function findLibraryItemForVariant(library: LibraryItem[], variant: ModelVariant): LibraryItem | null {`
- Line 202: `findCatalogVariantForLibraryItem` — `function findCatalogVariantForLibraryItem(families: ModelFamily[], item: LibraryItem): ModelVariant | null {`
- Line 211: `defaultVariantForFamily` — `function defaultVariantForFamily(family: ModelFamily | null | undefined): ModelVariant | null {`
- Line 218: `findVariantById` — `function findVariantById(families: ModelFamily[], variantId: string | null | undefined): ModelVariant | null {`
- Line 231: `firstDirectVariant` — `function firstDirectVariant(families: ModelFamily[]): ModelVariant | null {`
- Line 238: `findVariantForReference` — `function findVariantForReference(`
- Line 262: `titleFromPrompt` — `function titleFromPrompt(prompt: string) {`
- Line 266: `signedDelta` — `function signedDelta(value: number, digits = 1, suffix = "") {`
- Line 306: `settingsDraftFromWorkspace` — `function settingsDraftFromWorkspace(settings: AppSettings) {`
- Line 379: `setError` — `function setError(msg: string | null) {`
- Line 399: `refreshWorkspace` — `async function refreshWorkspace(preferredChatId?: string) {`
- Line 677: `enabledDirectoryCount` — `const enabledDirectoryCount = (workspace.settings?.modelDirectories ?? []).filter((directory) => directory.enabled).length;`
- Line 874: `connect` — `function connect(base: string) {`
- Line 913: `handleServerLogScroll` — `function handleServerLogScroll() {`
- Line 920: `scrollServerLogToBottom` — `function scrollServerLogToBottom() {`
- Line 933: `updateConversionDraft<K extends keyof typeof conversionDraft>` — `function updateConversionDraft<K extends keyof typeof conversionDraft>(key: K, value: (typeof conversionDraft)[K]) {`
- Line 940: `updateLaunchSetting<K extends keyof LaunchPreferences>` — `function updateLaunchSetting<K extends keyof LaunchPreferences>(key: K, value: LaunchPreferences[K]) {`
- Line 947: `prepareCatalogConversion` — `function prepareCatalogConversion(model: ModelVariant) {`
- Line 957: `prepareLibraryConversion` — `function prepareLibraryConversion(item: LibraryItem) {`
- Line 972: `loadPayloadFromVariant` — `function loadPayloadFromVariant(variant: ModelVariant, nextTab?: TabId) {`
- Line 993: `sessionModelPayload` — `function sessionModelPayload(session?: ChatSession | null) {`
- Line 1023: `mergeSessionMetadata` — `function mergeSessionMetadata(session: ChatSession, patch: Partial<ChatSession>): ChatSession {`
- Line 1027: `persistSessionChanges` — `async function persistSessionChanges(sessionId: string, patch: Partial<ChatSession>) {`
- Line 1058: `handleConvertModel` — `async function handleConvertModel() {`
- Line 1099: `handleLoadModel` — `async function handleLoadModel(payload: {`
- Line 1164: `handleLoadLibraryItem` — `async function handleLoadLibraryItem(item: LibraryItem, nextTab: TabId) {`
- Line 1175: `openModelSelector` — `function openModelSelector(action: "chat" | "server" | "thread", preselectedKey?: string) {`
- Line 1180: `confirmLaunch` — `async function confirmLaunch(selectedKey: string) {`
- Line 1237: `handleRevealPath` — `async function handleRevealPath(path: string) {`
- Line 1258: `handleUnloadModel` — `async function handleUnloadModel() {`
- Line 1271: `handleCreateSession` — `async function handleCreateSession() {`
- Line 1304: `handleRenameActiveThread` — `async function handleRenameActiveThread() {`
- Line 1316: `handleToggleThreadPin` — `async function handleToggleThreadPin(session: ChatSession) {`
- Line 1323: `handleSelectThreadModel` — `async function handleSelectThreadModel(nextKey: string) {`
- Line 1338: `handleLoadActiveThreadModel` — `async function handleLoadActiveThreadModel() {`
- Line 1354: `handleSelectServerModel` — `function handleSelectServerModel(nextKey: string) {`
- Line 1358: `handleLoadServerModel` — `async function handleLoadServerModel() {`
- Line 1373: `updateBenchmarkDraft<K extends keyof BenchmarkRunPayload>` — `function updateBenchmarkDraft<K extends keyof BenchmarkRunPayload>(key: K, value: BenchmarkRunPayload[K]) {`
- Line 1380: `handleRunBenchmark` — `async function handleRunBenchmark() {`
- Line 1418: `handleAddDirectory` — `function handleAddDirectory() {`
- Line 1440: `handleToggleDirectory` — `function handleToggleDirectory(directoryId: string) {`
- Line 1449: `handleRemoveDirectory` — `function handleRemoveDirectory(directoryId: string) {`
- Line 1456: `handleSaveSettings` — `async function handleSaveSettings() {`
- Line 1492: `handleStopServer` — `async function handleStopServer() {`
- Line 1521: `handleRestartServer` — `async function handleRestartServer() {`
- Line 1569: `threadPatchFromVariant` — `function threadPatchFromVariant(variant: ModelVariant): Pick<`
- Line 1585: `handleApplyVariantToActiveThread` — `async function handleApplyVariantToActiveThread(variant: ModelVariant) {`
- Line 1593: `handleStartThreadWithVariant` — `async function handleStartThreadWithVariant(variant: ModelVariant) {`
- Line 1625: `sendMessage` — `async function sendMessage() {`
- Line 1756: `renderCapabilityIcons` — `function renderCapabilityIcons(capabilities: string[], max = 5) {`
- Line 1783: `activePresetId` — `function activePresetId(settings: LaunchPreferences): string | null {`
- Line 1793: `applyPreset` — `function applyPreset(presetId: string) {`
- Line 1799: `renderSliderField` — `function renderSliderField(props: {`
- Line 1842: `renderLaunchControls` — `function renderLaunchControls() {`
- Line 1947: `renderLaunchProfileEditor` — `function renderLaunchProfileEditor(title: string, description: string) {`
- Line 1959: `renderLaunchModal` — `function renderLaunchModal() {`
- Line 1969: `setSelectedLaunchKey` — `const setSelectedLaunchKey = (key: string) => setPendingLaunch((prev) => prev ? { ...prev, preselectedKey: key } : null);`
- Line 2059: `renderDashboard` — `function renderDashboard() {`
- Line 2171: `renderOnlineModels` — `function renderOnlineModels() {`
- Line 2354: `toggleLibrarySort` — `function toggleLibrarySort(key: "name" | "format" | "size" | "modified") {`
- Line 2363: `sortIndicator` — `function sortIndicator(key: string) {`
- Line 2368: `renderMyModels` — `function renderMyModels() {`
- Line 2457: `renderConversion` — `function renderConversion() {`
- Line 2717: `renderChat` — `function renderChat() {`
- Line 2906: `copyText` — `function copyText(text: string) {`
- Line 2910: `renderServer` — `function renderServer() {`
- Line 3133: `renderBenchmarks` — `function renderBenchmarks() {`
- Line 3409: `renderLogs` — `function renderLogs() {`
- Line 3450: `renderSettings` — `function renderSettings() {`

### `src/api.test.ts`

- Exported symbols found: 0
- Function-like declarations found: 0
- Classes found: 0

### `src/api.ts`

- Exported symbols found: 0
- Function-like declarations found: 5
- Classes found: 0

#### Function-like declarations

- Line 28: `resetBackendRuntimeCache` — `function resetBackendRuntimeCache() {`
- Line 56: `fetchJson<T>` — `async function fetchJson<T>(path: string): Promise<T> {`
- Line 74: `postJson<T>` — `async function postJson<T>(path: string, body?: object): Promise<T> {`
- Line 78: `patchJson<T>` — `async function patchJson<T>(path: string, body?: object): Promise<T> {`
- Line 82: `sendJson<T>` — `async function sendJson<T>(method: "POST" | "PATCH", path: string, body?: object): Promise<T> {`

### `src/components/Panel.tsx`

- Exported symbols found: 1
- Function-like declarations found: 0
- Classes found: 0

#### Exports

- Line 10: `function Panel`

### `src/components/PerformancePreview.tsx`

- Exported symbols found: 1
- Function-like declarations found: 3
- Classes found: 0

#### Exports

- Line 30: `function PerformancePreview`

#### Function-like declarations

- Line 11: `fmt` — `function fmt(value: number, digits = 1): string {`
- Line 15: `getFitStatus` — `function getFitStatus(optimizedCacheGb: number, diskSizeGb: number, availableGb: number) {`
- Line 23: `getSpeedLabel` — `function getSpeedLabel(tokS: number): { label: string; className: string } | null {`

### `src/components/ProgressRow.tsx`

- Exported symbols found: 1
- Function-like declarations found: 0
- Classes found: 0

#### Exports

- Line 12: `function ProgressRow`

### `src/components/StatCard.tsx`

- Exported symbols found: 1
- Function-like declarations found: 0
- Classes found: 0

#### Exports

- Line 7: `function StatCard`

### `src/main.tsx`

- Exported symbols found: 0
- Function-like declarations found: 0
- Classes found: 0

### `src/mockData.ts`

- Exported symbols found: 1
- Function-like declarations found: 0
- Classes found: 0

#### Exports

- Line 3: `const mockWorkspace`

### `src/types.ts`

- Exported symbols found: 0
- Function-like declarations found: 0
- Classes found: 0

### `src/vite-env.d.ts`

- Exported symbols found: 0
- Function-like declarations found: 0
- Classes found: 0

## Native desktop backend (Rust)

### `src-tauri/src/lib.rs`

- Structs: 6
- Enums: 0
- Impl blocks: 2
- Functions: 38

#### Structs

- Line 23: `BackendRuntimeInfo`
- Line 55: `EmbeddedRuntimeManifest`
- Line 68: `SavedDesktopSettings`
- Line 75: `EmbeddedRuntime`
- Line 87: `ManagedBackend`
- Line 93: `BackendManager`

#### Impl blocks

- Line 36: `impl Default`
- Line 120: `impl BackendManager`

#### Functions

- Line 37: `default` — `fn default() -> Self {`
- Line 98: `app_version` — `fn app_version() -> String {`
- Line 103: `backend_runtime_info` — `fn backend_runtime_info(state: State<'_, BackendManager>) -> BackendRuntimeInfo {`
- Line 108: `stop_backend_sidecar` — `fn stop_backend_sidecar(state: State<'_, BackendManager>) -> BackendRuntimeInfo {`
- Line 114: `restart_backend_sidecar` — `fn restart_backend_sidecar(app: AppHandle, state: State<'_, BackendManager>) -> BackendRuntimeInfo {`
- Line 121: `bootstrap` — `fn bootstrap(&self, app: &AppHandle) {`
- Line 253: `runtime_info` — `fn runtime_info(&self) -> BackendRuntimeInfo {`
- Line 292: `shutdown` — `fn shutdown(&self) {`
- Line 304: `apply_embedded_runtime_env` — `fn apply_embedded_runtime_env(command: &mut Command, runtime: &EmbeddedRuntime) {`
- Line 330: `resolve_cert_bundle` — `fn resolve_cert_bundle(runtime: &EmbeddedRuntime) -> Option<PathBuf> {`
- Line 338: `apply_library_path` — `fn apply_library_path(command: &mut Command, variable: &str, entries: &[PathBuf]) {`
- Line 344: `join_paths` — `fn join_paths(entries: &[PathBuf]) -> Option<OsString> {`
- Line 351: `prepend_env_paths` — `fn prepend_env_paths(variable: &str, entries: &[PathBuf]) -> Option<OsString> {`
- Line 362: `source_workspace_root` — `fn source_workspace_root() -> PathBuf {`
- Line 367: `current_platform_tag` — `fn current_platform_tag() -> String {`
- Line 375: `embedded_debug_enabled` — `fn embedded_debug_enabled() -> bool {`
- Line 379: `debug_embedded` — `fn debug_embedded(message: impl AsRef<str>) {`
- Line 385: `embedded_resource_roots` — `fn embedded_resource_roots(app: &AppHandle) -> Vec<PathBuf> {`
- Line 402: `resolve_embedded_runtime` — `fn resolve_embedded_runtime(app: &AppHandle) -> Option<EmbeddedRuntime> {`
- Line 487: `ensure_embedded_runtime_extracted` — `fn ensure_embedded_runtime_extracted(`
- Line 547: `legacy_resource_python_root` — `fn legacy_resource_python_root(app: &AppHandle) -> Option<PathBuf> {`
- Line 555: `resolve_workspace_root` — `fn resolve_workspace_root(app: &AppHandle) -> Option<PathBuf> {`
- Line 575: `resolve_python_executable` — `fn resolve_python_executable(workspace_root: &Path) -> Option<PathBuf> {`
- Line 599: `resolve_llama_server` — `fn resolve_llama_server(workspace_root: &Path) -> Option<PathBuf> {`
- Line 621: `resolve_llama_cli` — `fn resolve_llama_cli(workspace_root: &Path) -> Option<PathBuf> {`
- Line 643: `resolve_candidate` — `fn resolve_candidate(value: impl Into<PathBuf>) -> Option<PathBuf> {`
- Line 654: `find_in_path` — `fn find_in_path(names: &[&str]) -> Option<PathBuf> {`
- Line 674: `open_log_file` — `fn open_log_file(path: &Path) -> Option<std::fs::File> {`
- Line 682: `read_log_tail` — `fn read_log_tail(path: &Path) -> String {`
- Line 689: `settings_path` — `fn settings_path() -> Option<PathBuf> {`
- Line 693: `saved_backend_port` — `fn saved_backend_port() -> Option<u16> {`
- Line 702: `saved_allow_remote_connections` — `fn saved_allow_remote_connections() -> Option<bool> {`
- Line 709: `saved_auto_start_server` — `fn saved_auto_start_server() -> Option<bool> {`
- Line 716: `selected_bind_host` — `fn selected_bind_host(allow_remote_connections: bool) -> &'static str {`
- Line 724: `select_backend_port` — `fn select_backend_port(preferred: u16, allow_remote_connections: bool) -> u16 {`
- Line 735: `port_responding` — `fn port_responding(port: u16) -> bool {`
- Line 739: `wait_for_port` — `fn wait_for_port(port: u16, timeout: Duration) -> bool {`
- Line 750: `run` — `pub fn run() {`

### `src-tauri/src/main.rs`

- Structs: 0
- Enums: 0
- Impl blocks: 0
- Functions: 1

#### Functions

- Line 3: `main` — `fn main() {`

## Limitations

- Python signatures are AST-derived and usually accurate.
- TS/TSX/MJS and Rust sections are source inventories built with lightweight matching.
- Large React components may contain anonymous inline callbacks that are not named individually.
- This is an inventory README, not a hand-written behavioral spec.
