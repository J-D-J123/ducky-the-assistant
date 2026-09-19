"""
Live mic -> text with Moonshine Voice (CPU friendly, streaming).

    pip install moonshine-voice

First run downloads the default English model, later runs work offline.
"""
import queue
import shutil
import threading
import time

from moonshine_voice import MicTranscriber

lines_q = queue.Queue()  # finished lines waiting to go to your model


def send_to_model(text: str) -> None:
    """Replace this with your model call (API request, local LLM, etc.)."""
    print(f"[-> model] {text}", flush=True)


def model_worker() -> None:
    # runs on its own thread so a slow model call never blocks the mic
    while True:
        text = lines_q.get()
        if text is None:
            break
        try:
            send_to_model(text)
        except Exception as e:
            print(f"\n[model error] {e}", flush=True)


def show_partial(text: str) -> None:
    # live, in-progress text that keeps rewriting itself on one line
    width = shutil.get_terminal_size((100, 20)).columns - 1
    print("\r" + text[-width:].ljust(width), end="", flush=True)


def handle_line(line) -> None:
    # fires once when you pause and the line is final
    text = line.text.strip()
    width = shutil.get_terminal_size((100, 20)).columns - 1
    print("\r" + " " * width + "\r", end="")  # clear the partial
    if text:
        print(text, flush=True)
        lines_q.put(text)


def main() -> None:
    worker = threading.Thread(target=model_worker, daemon=True)
    worker.start()

    mic = (
        MicTranscriber()
        .language("en")
        .on_text(show_partial)
        .on_line(handle_line)
    )
    mic.load()   # blocks on first run while the model downloads
    mic.start()
    print("Listening... Speak naturally. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()    # completes any active line before returning
        mic.close()
        lines_q.put(None)
        worker.join()
        print("\nStopped.")


if __name__ == "__main__":
    main()
