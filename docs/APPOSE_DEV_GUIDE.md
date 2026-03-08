# Appose Developer Guide

Developer reference for writing Python code that runs inside the Appose embedded Python environment. This documents constraints, pitfalls, and required patterns that differ from standalone Python scripts.

## What is Appose?

[Appose](https://github.com/apposed/appose) provides shared-memory IPC for running Python code embedded within the Java process. The extension uses Appose to run all training, inference, and pretraining workloads without requiring a separate Python server process.

Key difference from normal Python: **your code runs inside an embedded interpreter managed by Java, not as a standalone process.** This has implications for multiprocessing, JSON serialization, error handling, and progress reporting.

## Constraint 1: No Multiprocessing (num_workers=0)

**Rule: Always use `num_workers=0` in PyTorch DataLoaders.**

```python
# WRONG -- will silently deadlock on Windows
dataloader = DataLoader(dataset, batch_size=8, num_workers=4)

# CORRECT
dataloader = DataLoader(dataset, batch_size=8, num_workers=0)
```

**Why:** Python multiprocessing spawns child processes that need to import the `__main__` module. In embedded Python (Appose), there is no proper `__main__` module and no `if __name__ == '__main__':` guard. On Windows (which uses `spawn` rather than `fork`), this causes a **silent deadlock** -- the DataLoader hangs indefinitely at the first iteration with no error, no exception, and no log message.

**Symptoms when violated:**
- Progress stuck at "Starting first epoch..." with no GPU utilization
- Elapsed timer increases but nothing else happens
- No error in logs -- the hang is completely silent
- Only diagnosable by checking Python Console or noticing GPU is idle

**History:** This bug was discovered in MAE pretraining (2026-03-07) where `num_workers=min(4, os.cpu_count())` caused an indefinite hang. The training service already had `num_workers=0` but the MAE service was written independently without this knowledge.

## Constraint 2: JSON Serialization -- No NaN or Infinity

**Rule: Always sanitize float values before `json.dumps()` in progress callbacks.**

```python
import math

# WRONG -- NaN produces invalid JSON that breaks ALL progress updates
task.update(message=json.dumps({"loss": float('nan')}))

# CORRECT
safe_loss = loss if math.isfinite(loss) else 0.0
task.update(message=json.dumps({"loss": safe_loss}))
```

**Why:** Python's `json.dumps()` serializes `float('nan')` as the bare token `NaN` and `float('inf')` as `Infinity`. These are **not valid JSON** (JSON only allows `null`, `true`, `false`, numbers, strings, arrays, objects). On the Java side, Gson's `JsonParser.parseString()` throws `JsonSyntaxException`, and because the progress listener catches exceptions, the update is **silently dropped**. One NaN value causes the entire progress message to be lost.

**Standard pattern for progress callbacks:**

```python
import math

def _safe(v):
    """Sanitize a value for JSON serialization."""
    return v if isinstance(v, (int, str)) or (
        isinstance(v, float) and math.isfinite(v)
    ) else 0.0

def _safe_dict(d):
    """Sanitize all values in a dict for JSON serialization."""
    return {k: _safe(v) for k, v in d.items()} if d else {}

# Use in callbacks:
task.update(message=json.dumps({
    "train_loss": _safe(train_loss),
    "per_class_iou": _safe_dict(per_class_iou),
}))
```

**When NaN/Inf can occur:**
- Training loss when gradients explode (diverging training)
- Learning rate during warmup edge cases
- Division by zero in metric calculations (e.g., IoU with empty classes)
- Per-class loss when a class has zero pixels in a batch

**Symptoms when violated:**
- Progress monitor stuck showing the last successfully parsed update
- Status bar shows a stale message (e.g., setup phase name)
- No errors visible at default log level (parse failure logged at WARN)
- Training appears to run (GPU active) but UI never updates

**History:** This bug caused a 6-hour MAE pretraining run to show nothing but "starting_training" (2026-03-07). The NaN came from the reconstruction loss during the first few batches.

## Constraint 3: Progress Reporting Pattern

All Appose scripts communicate with Java through the `task` object injected into the script scope.

### Setup phases

Report setup progress before the training loop:

```python
def setup_callback(phase, data=None):
    msg = {
        "status": "setup",        # Triggers isSetupPhase() on Java side
        "setup_phase": phase,      # Machine-readable phase name
        "epoch": 0,
        "total_epochs": total_epochs,
    }
    task.update(message=json.dumps(msg), current=0, maximum=total_epochs)
```

Standard phase names and their Java-side display strings:
- `"loading_data"` -> "Loading image tiles..."
- `"computing_stats"` -> "Computing normalization statistics..."
- `"creating_model"` -> "Creating model architecture..."
- `"starting_training"` -> "Starting first epoch..."
- `"saving_model"` -> "Saving encoder weights..."

### Epoch progress

Report after each epoch completes:

```python
task.update(
    message=json.dumps({
        "epoch": epoch,
        "total_epochs": total,
        "train_loss": _safe(loss),
        # ... other metrics
    }),
    current=epoch,
    maximum=total
)
```

Note: epoch progress messages must NOT include `"status": "setup"` or `"status": "initializing"` -- the absence of these keys is how Java distinguishes epoch updates from setup phases.

### Cancellation

Bridge the Appose cancellation mechanism to a threading.Event:

```python
import threading, time

cancel_flag = threading.Event()

def watch_cancel():
    while not cancel_flag.is_set():
        if task.cancel_requested:
            cancel_flag.set()
            break
        time.sleep(0.5)

cancel_watcher = threading.Thread(target=watch_cancel, daemon=True)
cancel_watcher.start()

# Pass cancel_flag to your service, check it in training loops
```

### Outputs

Set task outputs before the script exits:

```python
task.outputs["status"] = "completed"
task.outputs["model_path"] = "/path/to/model.pt"
```

## Constraint 4: Error Visibility

**Rule: Log errors at WARNING or higher, never DEBUG.**

```python
# WRONG -- failures invisible in QuPath log
except Exception as e:
    logger.debug("Failed: %s", e)

# CORRECT
except Exception as e:
    logger.warning("Failed: %s", e)
```

On the Java side, Appose event listeners catch exceptions to prevent crashes. If the error is logged at DEBUG, it is invisible in the default QuPath log view. Users see no indication of what went wrong.

## Constraint 5: Globals Available in Script Scope

Appose scripts receive these injected globals (set up by `init_shared.py`):

| Variable | Type | Description |
|----------|------|-------------|
| `task` | `appose.Task` | Progress reporting and output |
| `inference_service` | `InferenceService` | Shared inference service instance |
| `gpu_manager` | `GPUManager` | GPU device and memory management |

Plus any `inputs` specified when creating the task (e.g., `config`, `data_path`).

Check `inference_service is not None` at script start to verify initialization succeeded.

## Constraint 6: Threading

**Rule: Use `threading` (not `multiprocessing`) for concurrent work.**

The Appose environment supports Python threading but not multiprocessing (see Constraint 1). Background threads work fine for cancellation watchers and similar lightweight tasks.

## Checklist for New Appose Scripts

Before submitting a new `.py` script in `src/main/resources/.../scripts/`:

- [ ] **DataLoader `num_workers=0`** -- never use multiprocessing workers
- [ ] **NaN/Inf guards** on all float values passed to `json.dumps()`
- [ ] **`task.update()` wrapped in try/except** to prevent serialization errors from crashing the script
- [ ] **Setup phases reported** so users see progress during initialization
- [ ] **Cancellation bridge** with `threading.Event` and daemon watcher thread
- [ ] **Error logging at WARN+** not DEBUG, for all catch blocks
- [ ] **No bare `import` of heavy libraries at top level** -- import inside functions or after the initialization check to keep startup fast
- [ ] **Test on Windows** -- encoding, file paths (backslashes), and multiprocessing all behave differently

## Checklist for New Python Services

Before submitting a new service in `python_server/dlclassifier_server/services/`:

- [ ] **DataLoader `num_workers=0`** -- consistent with all other services
- [ ] **Progress callback wrapped in try/except** -- never let a callback error kill training
- [ ] **Diagnostic logging at key milestones** -- dataset size, model params, device, first batch shape, first loss value
- [ ] **Division-by-zero guards** in metric calculations
- [ ] **GPU memory cleanup** via `gpu_manager.clear_cache()` in finally blocks
