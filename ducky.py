"""
Talk to a local Ollama model with your voice and hear it answer (Kokoro TTS, CPU, free).

    pip install moonshine-voice kokoro-onnx piper-tts sounddevice numpy
    ollama serve            # usually already running as a service

Speak -> Moonshine transcribes -> when you really stop talking, the line goes to Ollama
-> the reply streams to the terminal AND is spoken while it's still being written.

Interrupting:
  * Just start talking over it and it stops (works best with headphones).
  * Say "stop" / "wait" / "hold on" to pause it without asking anything new.
  * Say "continue" / "resume" / "go on" and it picks its answer back up where you cut it off.
  * Keyboard backup: Enter = interrupt, "c" + Enter = continue, or just type a message.

The two Kokoro model files are downloaded next to this script on first run.
"""
import collections
import difflib
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import sounddevice as sd
from datetime import datetime
from kokoro_onnx import Kokoro
from moonshine_voice import MicTranscriber

# ---------------- config ----------------
OLLAMA_URL = "http://localhost:11434"
# MODEL = "gemma3:4b-it-qat"       # must match a name from `ollama list`
MODEL = "llama3.1:8b"
SYSTEM_PROMPT = (                # set to None to use the model's own built-in prompt instead
    "You're a relaxed, friendly voice assistant. Talk naturally, keep replies short, "
    "never use emojis or markdown since your words are spoken aloud, and don't swear unless I do."
)
KEEP_ALIVE = "30m"               # keep the LLM loaded so replies start fast
MAX_HISTORY = 10                 # messages of context sent each turn (lower = faster on a Pi)

# voice: am_michael (default), am_fenrir, am_puck, am_adam, am_eric, am_liam, am_onyx
#        british: bm_george, bm_lewis, bm_daniel, bm_fable
VOICE = "am_michael"
SPEED = 1.1                      # 1.0 = normal, 0.9 = calmer, 1.2 = brisk
KOKORO_MODEL = "kokoro-v1.0.int8.onnx"  # ~2x faster than "kokoro-v1.0.onnx" (full quality, slower)
TTS_ENGINE = "auto"              # "auto": Kokoro if this CPU runs it fast enough, otherwise Piper. Or force "kokoro" / "piper"
AUTO_MAX_RTF = 0.6               # auto mode keeps Kokoro only if it makes speech at least this fast (0.6 = 1.7x real time)
PIPER_VOICE = "en_US-ryan-medium"  # male. Others: en_US-joe-medium, en_US-hfc_male-medium, en_GB-alan-medium
BENCH_TEXT = "Hey there! I'm a pretty simple voice assistant, just here to chat."
SHOW_TTS_TIMING = False          # True prints how long each chunk took to generate vs its length
AUDIO_LATENCY = 0.2              # seconds of speaker buffer: lower = snappier, higher = fewer clicks
CPU_SPLIT = False                # give the LLM and the voice their own CPU cores so they stop fighting
SHOW_LATENCY = True              # prints a one-line breakdown whenever it takes >2s to start talking

# how speech is fed out while the reply is still being written
LEAD_TARGET_S = 2.0              # keep about this much audio queued ahead of what you're hearing
CLAUSE_LEAD_S = 1.0              # if the queue runs lower than this, don't wait for a full sentence
SECS_PER_WORD = 0.33             # rough speaking pace, only used to estimate queued audio
FIRST_MIN_WORDS = 1              # first chunk: starts at the first period/!/? (a comma needs 2+ words)
FIRST_FORCE_WORDS = 4            # ...or after this many words even with no punctuation
MIN_WORDS = 3                    # later chunks: at least this many words
FORCE_WORDS = 12                 # ...or cut here (only if the queue is nearly empty)
MAX_CHUNK_WORDS = 30             # never make one chunk longer than this

# True  = headphones: any speech from you cuts it off (most reliable, use this if you can).
# False = speakers: it tells your voice from its own by which words are NEW, so it can
#         still be interrupted, but it needs a few words to be sure.
HEADPHONES = False
INTERRUPT_MIN_WORDS = 3          # speaker mode: new words needed before it decides you're cutting in
SPEAK_GRACE_S = 1.0              # speaker mode: extra echo-filtering after it stops talking
# words that cut it off instantly (includes what the recognizer often mishears "wait"/"pause" as)
STOP_WORDS = {"stop", "wait", "pause", "hold", "hang", "shush", "shh", "quiet", "enough",
              "weight", "paws", "pours"}
INTERRUPT_TAIL = 5               # speaker mode: only the last few words of a partial are judged
ECHO_MATCH_WINDOW_S = 10.0       # speaker mode: for this long after it talks, lines that sound like
ECHO_FRACTION = 0.6              #   what it just said (this fraction of the key words) are thrown away
# small everyday words: they never count as proof that you (rather than its echo) are talking
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "so", "if", "of", "to", "in", "on", "at", "for", "with", "by",
    "from", "as", "is", "are", "was", "were", "be", "been", "am", "it", "it's", "its", "that", "that's",
    "this", "these", "those", "i", "i'm", "i've", "i'll", "i'd", "you", "you're", "you've", "you'll",
    "we", "we're", "he", "she", "they", "they're", "them", "his", "her", "my", "your", "our", "me", "us",
    "do", "does", "did", "don't", "just", "not", "no", "yes", "yeah", "yep", "oh", "well", "like",
    "then", "than", "there", "there's", "here", "what", "what's", "who", "how", "when", "where", "why",
    "can", "could", "will", "would", "should", "have", "has", "had", "about", "up", "out", "get", "got",
    "really", "very", "some", "any", "all", "one", "okay", "ok",
}

# interrupt by mic LOUDNESS: needs no speech recognition, so it works even when the
# recognizer garbles short words like "wait" or "pause" while the bot is talking
VOICE_INTERRUPT = True
VOICE_RATIO = 2.2                # you must be this many times louder than its echo (speaker mode)
VOICE_MIN_RMS = 0.02             # ...and at least this loud. Raise it if noise cuts it off
SHOW_DEBUG = False               # True prints what the mic hears while it talks and why lines get dropped

# turn-taking: how long it waits after you go quiet before it answers
END_OF_TURN_S = 0.8              # normal sentence
QUESTION_WAIT_S = 0.5            # you ended on a "?", so it's probably its turn
TRAIL_EXTRA_S = 1.2              # extra wait when you trail off ("and...", "so", "um")
HOLD_FALLBACK_S = 6.0            # safety: answer anyway if a line never finishes
FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "eh", "hm", "hmm", "mm", "mhm"}
TRAIL_WORDS = FILLERS | {
    "and", "but", "so", "or", "because", "like", "the", "a", "an", "to", "of",
    "that", "then", "with", "if", "when", "which", "also", "well", "i", "my",
}
FILLER_RE = re.compile(r"\b(?:u+m+|u+h+|e+r+m*|h+m+|m+h*m+)\b[,.]?\s*", re.I)

# voice commands (matched against your whole message, punctuation ignored)
_LEAD = r"(?:(?:ok|okay|yeah|yes|yep|sure|please|and|so|alright|hey)\s+)*"
RESUME_RE = re.compile(
    r"^" + _LEAD + r"(?:you can\s+)?(?:continue|resume|go on|keep going|carry on|go ahead|"
    r"keep talking|finish that|finish your thought|finish your answer|"
    r"finish what you were saying|where were we|where you left off)"
    r"(?:\s+(?:please|now|then|again|talking))?$"
)
STOP_FILLER = {"on", "up", "a", "one", "sec", "second", "moment", "minute", "just", "please",
               "ok", "okay", "hey", "for", "be", "talking", "it", "that", "now", "right", "give", "me"}
HOLD_RE = re.compile(
    r"^(?:(?:ok|okay|hey|please)\s+)*(?:just\s+)?(?:a|one)\s+(?:sec|second|moment|minute)(?:\s+please)?$"
)
# ----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RELEASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
VOICES_FILE = "voices-v1.0.bin"
LANG = "en-gb" if VOICE.startswith("b") else "en-us"

_cores = os.cpu_count() or 4
TTS_THREADS = max(2, _cores // 2) if CPU_SPLIT else None
LLM_THREADS = max(2, _cores - TTS_THREADS) if CPU_SPLIT else None

lines_q = queue.Queue()          # (text_to_send, original_question) from the mic
tts_q = queue.Queue()            # (epoch, text, est_seconds) waiting to be synthesized
play_q = queue.Queue()           # (epoch, samples, sample_rate, text) waiting to be played
generating = threading.Event()   # set while the LLM is replying
cancel = threading.Event()       # set when you cut in mid-reply
barge_in = threading.Event()     # set once you've interrupted the current reply
history = []
history_lock = threading.Lock()
interrupt_lock = threading.Lock()

epoch = 0                        # bumped on barge-in so stale audio gets dropped
epoch_lock = threading.Lock()
last_audio_end = 0.0
last_write_at = 0.0             # when audio was last sent to the speaker
last_reply_words = set()         # words the AI has said this reply, used to spot echo
last_reply_seq = []              # the same words in the order it said them
spoken_log = []                  # chunks that were fully played this reply
current_question = ""            # what the AI is currently answering
last_assistant_msg = None        # this reply's entry in history (once it has finished writing)
pending_resume = None            # {"question", "spoken"} for a reply you cut off
active_stream = None             # TextStream for the reply being spoken

horizon = 0.0                    # clock time until which audio is already queued up
voice_sr = 24000                 # sample rate of whichever voice engine is in use
reply_t0 = 0.0                   # latency bookkeeping for the current reply
t_first_chunk = 0.0
t_audio1 = 0.0
first_synth_s = 0.0
last_rtf = 0.0
horizon_lock = threading.Lock()

turn_lock = threading.Lock()
turn_parts = []                  # your finished lines, waiting to see if you're really done
turn_timer = None
last_partial = ""
line_open = False                # a line is in progress (we've seen its first partial)
line_tainted = False             # ...and it began while the AI was talking, so it contains echo

WAKE_WORD = "ducky"
WAKE_TIMEOUT_S = 5.0
wake_armed_until = 0.0

WAKE_RE = re.compile(
    r"^\s*(?:hey\s+)?ducky\b[\s,:;-]*(.*)$",
    re.IGNORECASE
)


def extract_wake_command(text: str):
    """
    Returns:
        ("command", command_text) if wake word + command were spoken
        ("wake_only", "") if only 'ducky' / 'hey ducky' was spoken
        (None, None) if no wake word was used
    """
    match = WAKE_RE.match(text)
    if not match:
        return None, None

    command = match.group(1).strip()

    if command:
        return "command", command

    return "wake_only", ""


# ---------- helpers ----------
def term_width() -> int:
    return shutil.get_terminal_size((100, 20)).columns - 1


def machine_time_context() -> str:
    now = datetime.now().astimezone()
    return (
        f"The computer's current local date and time is "
        f"{now.strftime('%A, %B %d, %Y %I:%M:%S %p %Z')}. "
        f"Use this as the authoritative current time when I ask about the time, "
        f"date, day, or time of day."
    )


def current_epoch() -> int:
    with epoch_lock:
        return epoch


def tokens(text: str):
    return re.findall(r"[\w']+", text.lower().replace("\u2019", "'").replace("\u2018", "'"))


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def ai_active() -> bool:
    s = active_stream
    return (
        generating.is_set()
        or tts_q.unfinished_tasks > 0
        or play_q.unfinished_tasks > 0
        or (s is not None and s.ep == current_epoch() and s.has_text())
    )


def in_echo_window() -> bool:
    return ai_active() or time.time() < last_audio_end + SPEAK_GRACE_S


def _in_reply(word: str) -> bool:
    """Is this word (or something that sounds a lot like it) in what the AI just said?"""
    if word in last_reply_words:
        return True
    if len(word) <= 4:
        return False  # short words too easily "sound like" others ("wait" vs "with")
    return bool(difflib.get_close_matches(word, list(last_reply_words), n=1, cutoff=0.75))


def _key_words(words):
    """Drop the everyday small words; what's left is what actually carries the meaning."""
    key = [w for w in words if w not in STOPWORDS]
    return key if key else words


def echo_score(text: str) -> float:
    """0..1: how much of this line is (a garbled version of) the AI's own last reply."""
    words = [w for w in tokens(text) if w not in FILLERS]
    if not words or not last_reply_words:
        return 0.0
    key = _key_words(words)
    return sum(_in_reply(w) for w in key) / len(key)


def longest_run(words) -> int:
    """Longest stretch of consecutive words that also appears, in the same order, in what it said."""
    if not last_reply_seq or not words:
        return 0
    vocab = list(last_reply_words)
    mapped = []
    for w in words:                                   # let "riding" match "writing"
        if w in last_reply_words or len(w) <= 4:
            mapped.append(w)
        else:
            close = difflib.get_close_matches(w, vocab, n=1, cutoff=0.75)
            mapped.append(close[0] if close else w)
    matcher = difflib.SequenceMatcher(None, mapped, last_reply_seq, autojunk=False)
    return max((blk.size for blk in matcher.get_matching_blocks()), default=0)


def tail_is_new(words) -> bool:
    """Your words land at the END of a line that starts with echo, so judge the end (key words only)."""
    tail = words[-INTERRUPT_TAIL:]
    key = [w for w in tail if w not in STOPWORDS]
    novel = [w for w in key if not _in_reply(w)]
    return len(novel) >= INTERRUPT_MIN_WORDS and len(novel) / len(key) >= 0.6


def is_user_speech(text: str) -> bool:
    """Does this sound like YOU talking, as opposed to the AI's voice coming back through the mic?"""
    words = [w for w in tokens(text) if w not in FILLERS]
    if not words:
        return False
    if any(w in STOP_WORDS and w not in last_reply_words for w in words):
        return True
    if HEADPHONES:
        return len(words) >= (2 if ai_active() else 1)
    if ai_active():
        return tail_is_new(words)
    key = _key_words(words)
    novel = [w for w in key if not _in_reply(w)]
    return len(novel) >= 1 and len(novel) / len(key) >= 0.6


def looks_like_echo(text: str, tainted: bool = False) -> bool:
    """
    Speaker mode: is this line just the AI's own voice coming back through the mic?
    `tainted` = the line started while it was talking (or right after), so it has echo in it.
    """
    words = [w for w in tokens(text) if w not in FILLERS]
    if ai_active():
        return not is_user_speech(text)               # while it talks, only clear new speech counts
    if not last_reply_words or not words:
        return False
    since = time.time() - last_audio_end
    if since < ECHO_MATCH_WINDOW_S and len(words) >= 2 and echo_score(text) >= ECHO_FRACTION:
        # short lines must also repeat its words IN ORDER ("you too" after "You too!"),
        # so a quick "help me" isn't mistaken for echo just because it said "help" somewhere
        if len(words) >= 4 or longest_run(words) >= 2:
            return True
    if tainted and len(words) >= 3 and echo_score(text) >= 0.3:
        # began during its speech AND partly sounds like it, with nothing clearly new at the end.
        # (A line with no overlap at all is just you talking quickly after it finished: keep it.)
        if not (any(w in STOP_WORDS and w not in last_reply_words for w in words) or tail_is_new(words)):
            return True
    return False


def ensure_file(name: str) -> str:
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        print(f"Downloading {name} (first run only)...", flush=True)
        urllib.request.urlretrieve(RELEASE_URL + name, path + ".part")
        os.replace(path + ".part", path)
    return path


def load_kokoro() -> Kokoro:
    model_path, voices_path = ensure_file(KOKORO_MODEL), ensure_file(VOICES_FILE)
    if TTS_THREADS:
        try:
            import onnxruntime as rt
            opts = rt.SessionOptions()
            opts.intra_op_num_threads = TTS_THREADS
            opts.inter_op_num_threads = 1
            session = rt.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            return Kokoro.from_session(session, voices_path)
        except Exception as e:
            print(f"(couldn't set the voice's CPU threads, using defaults: {e})", flush=True)
    return Kokoro(model_path, voices_path)


class KokoroEngine:
    name = "Kokoro"

    def __init__(self, kokoro):
        self.k = kokoro

    def create(self, text: str):
        return self.k.create(text, voice=VOICE, speed=SPEED, lang=LANG)


class PiperEngine:
    """Piper is much lighter than Kokoro: a bit less natural, but fast even on weak CPUs."""
    name = "Piper"

    def __init__(self, voice, syn_config):
        self.v = voice
        self.cfg = syn_config

    def create(self, text: str):
        parts, sr = [], int(self.v.config.sample_rate)
        for chunk in self.v.synthesize(text, syn_config=self.cfg):
            parts.append(np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1))
            sr = int(chunk.sample_rate)
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return audio, sr


def ensure_url(name: str, url: str) -> str:
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        print(f"Downloading {name} (first run only)...", flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def load_piper_engine() -> PiperEngine:
    from piper import PiperVoice, SynthesisConfig
    lang_region, speaker, quality = PIPER_VOICE.split("-", 2)      # en_US-ryan-medium
    base = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{lang_region.split('_')[0]}/{lang_region}/{speaker}/{quality}/{PIPER_VOICE}")
    model = ensure_url(PIPER_VOICE + ".onnx", base + ".onnx")
    ensure_url(PIPER_VOICE + ".onnx.json", base + ".onnx.json")
    return PiperEngine(PiperVoice.load(model), SynthesisConfig(length_scale=1.0 / SPEED))


def measure_rtf(engine):
    """How long it takes to make 1 second of speech (under 1.0 = faster than real time)."""
    engine.create("Ready.")                                 # warm-up
    t0 = time.time()
    samples, sr = engine.create(BENCH_TEXT)
    took = time.time() - t0
    return took / max(len(samples) / sr, 0.1), sr


def load_voice():
    """Pick the voice engine. Returns (engine, sample_rate)."""
    want = TTS_ENGINE.lower()
    if want in ("kokoro", "auto"):
        kokoro = KokoroEngine(load_kokoro())
        rtf, sr = measure_rtf(kokoro)
        print(f"Voice: Kokoro ({VOICE}) makes speech at {rtf:.2f}x real time.", flush=True)
        if want == "kokoro" or rtf <= AUTO_MAX_RTF:
            return kokoro, sr
        print(f"That's too slow for this CPU (want under {AUTO_MAX_RTF}), trying Piper...", flush=True)
        try:
            piper = load_piper_engine()
            prtf, psr = measure_rtf(piper)
            print(f"Voice: Piper ({PIPER_VOICE}) makes speech at {prtf:.2f}x real time.", flush=True)
            if prtf < rtf:
                return piper, psr
        except ImportError:
            print("Piper isn't installed. Run:  pip install piper-tts   (staying with Kokoro for now)", flush=True)
        except Exception as e:
            print(f"Couldn't start Piper ({e}), staying with Kokoro.", flush=True)
        return kokoro, sr
    piper = load_piper_engine()
    prtf, psr = measure_rtf(piper)
    print(f"Voice: Piper ({PIPER_VOICE}) makes speech at {prtf:.2f}x real time.", flush=True)
    return piper, psr


def clean_for_speech(text: str) -> str:
    text = (
        text.replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2014", ", ").replace("\u2013", "-")
    )
    text = text.replace("_", " ")
    # drops markdown symbols, emoji and other stuff a voice would mangle
    text = re.sub(r"[^\w\s.,!?;:'\"()\-%$&/+=]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------- feeding text to the voice ----------
def lead_s() -> float:
    """Roughly how many seconds of speech are already queued ahead of what you're hearing."""
    with horizon_lock:
        return max(0.0, horizon - time.time())


def add_lead(seconds: float) -> None:
    global horizon
    with horizon_lock:
        horizon = max(horizon, time.time()) + seconds


def fix_lead(delta: float) -> None:
    global horizon
    with horizon_lock:
        horizon += delta


def reset_lead() -> None:
    global horizon
    with horizon_lock:
        horizon = 0.0


TOKEN = re.compile(r"\S+\s*")
SENT_TOKEN = re.compile(r"[.!?][\"')\]]*\s*$")
CLAUSE_TOKEN = re.compile(r"[,;:][\"')\]]*\s*$")


def _sentence_end(tok: str) -> bool:
    return bool(SENT_TOKEN.search(tok)) or "\n" in tok[len(tok.rstrip()):]


def _clause_end(tok: str) -> bool:
    return bool(CLAUSE_TOKEN.search(tok))


class TextStream:
    """
    The reply as it streams in. The feeder thread pulls speakable chunks out of it,
    but only when the audio queue is running low, so chunks stay as long (and as
    natural-sounding) as possible without ever letting the speaker go quiet.
    """

    def __init__(self, ep: int):
        self.ep = ep
        self.buf = ""
        self.done = False
        self.emitted = 0
        self.lock = threading.Lock()

    def feed(self, piece: str) -> None:
        with self.lock:
            self.buf += piece

    def finish(self) -> None:
        with self.lock:
            self.done = True

    def has_text(self) -> bool:
        with self.lock:
            return bool(self.buf.strip())

    def take(self, lead: float):
        """The next chunk to speak, or None if it should wait."""
        with self.lock:
            self.buf = self.buf.lstrip()
            if not self.buf:
                return None
            if self.done:
                usable = self.buf
            else:
                # only whole words are safe to cut on while text is still arriving
                cut_at = max(self.buf.rfind(" "), self.buf.rfind("\n"))
                if cut_at <= 0:
                    return None
                usable = self.buf[:cut_at]
            toks = TOKEN.findall(usable)
            n = len(toks)
            if n == 0:
                return None

            first = self.emitted == 0
            min_w = FIRST_MIN_WORDS if first else MIN_WORDS
            force_w = FIRST_FORCE_WORDS if first else FORCE_WORDS
            overflow = n > MAX_CHUNK_WORDS
            if lead >= LEAD_TARGET_S and not overflow:
                return None  # plenty queued already, let the chunk grow
            limit = min(n, MAX_CHUNK_WORDS)

            cut = None
            for i in range(min_w - 1, limit):           # earliest sentence end
                if _sentence_end(toks[i]):
                    cut = i + 1
                    break
            if cut is None and (first or lead < CLAUSE_LEAD_S):
                for i in range(max(min_w, 2) - 1, limit):   # running low: a comma is good enough
                    if _clause_end(toks[i]):
                        cut = i + 1
                        break
            if cut is None and (overflow or (n >= force_w and (first or lead < CLAUSE_LEAD_S))):
                cut = limit
                for i in range(limit - 1, min_w - 2, -1):  # prefer ending on a comma
                    if _clause_end(toks[i]):
                        cut = i + 1
                        break
            if cut is None and self.done:
                cut = n
            if cut is None:
                return None

            chunk = "".join(toks[:cut])
            self.buf = self.buf[len(chunk):]
            self.emitted += 1
            return chunk.strip()


def feeder_worker() -> None:
    while True:
        s = active_stream
        if s is None or s.ep != current_epoch():
            time.sleep(0.02)
            continue
        chunk = s.take(lead_s())
        if chunk:
            speak(chunk, s.ep)
        else:
            time.sleep(0.015)


# ---------- speech pipeline ----------
def speak(text: str, ep: int) -> None:
    global t_first_chunk
    if ep != current_epoch():
        return
    if t_first_chunk == 0.0:
        t_first_chunk = time.time()
    words_said = tokens(text)
    last_reply_words.update(words_said)  # so its own voice can be recognised as echo
    last_reply_seq.extend(words_said)
    clean = clean_for_speech(text)
    if len(clean) < 2:
        return
    est = len(text.split()) * SECS_PER_WORD / SPEED
    add_lead(est)
    tts_q.put((ep, clean, est))


def stop_speech() -> None:
    """Drop everything queued; the player thread cuts off whatever is playing."""
    global epoch
    with epoch_lock:
        epoch += 1
    reset_lead()
    for q in (tts_q, play_q):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
            q.task_done()


def trim_silence(samples, sr: int, tail_s: float) -> np.ndarray:
    """Cut the dead air Kokoro leaves at both ends, keeping a short natural tail."""
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    if peak < 1e-4:
        return x
    loud = np.flatnonzero(np.abs(x) > max(0.003, 0.02 * peak))
    if loud.size == 0:
        return x
    start = max(0, int(loud[0]) - int(0.015 * sr))
    end = min(len(x), int(loud[-1]) + 1 + int(tail_s * sr))
    return x[start:end]


def synth_worker(engine) -> None:
    global first_synth_s, last_rtf
    while True:
        ep, text, est = tts_q.get()
        try:
            if ep == current_epoch():
                t0 = time.time()
                samples, sr = engine.create(text)
                # a real pause after . ! ? , a shorter one after a comma, none mid-phrase
                tail = 0.10 if SENT_TOKEN.search(text) else (0.04 if CLAUSE_TOKEN.search(text) else 0.0)
                audio = trim_silence(samples, sr, tail)
                length = len(audio) / sr
                took = time.time() - t0
                if first_synth_s == 0.0 and ep == current_epoch():
                    first_synth_s = took
                    last_rtf = took / max(length, 0.1)
                if SHOW_TTS_TIMING:
                    print(f"\n[tts] {took:.2f}s to make {length:.2f}s of audio "
                          f"({'OK' if took < length else 'TOO SLOW, will gap'})", flush=True)
                if ep == current_epoch():
                    fix_lead(length - est)
                    play_q.put((ep, audio, sr, text))
        except Exception as e:
            print(f"\n[tts error] {e}", flush=True)
        finally:
            tts_q.task_done()


def prepare_audio(samples) -> np.ndarray:
    """Mono float32 column with a few ms of fade at each end so chunks don't click."""
    audio = np.array(samples, dtype=np.float32).reshape(-1, 1)
    n = min(120, len(audio) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return np.ascontiguousarray(audio)


def note_first_audio() -> None:
    """Called when the first sound of a reply reaches the speaker; explains the wait if it was long."""
    global t_audio1
    now = time.time()
    t_audio1 = now
    total = now - reply_t0
    if SHOW_LATENCY and reply_t0 and total > 2.0:
        waiting = max(0.0, t_first_chunk - reply_t0)
        voice = first_synth_s
        other = max(0.0, total - waiting - voice)
        print(f"\n[lag] {total:.1f}s before it spoke = model's first words {waiting:.1f}s + "
              f"generating the voice {voice:.1f}s + other {other:.1f}s "
              f"(voice runs at {last_rtf:.1f}x real time; it needs to be under 1.0 to sound smooth)",
              flush=True)


def player_worker() -> None:
    """One output stream stays open, so there's no per-chunk open/close (the main source of pops/gaps)."""
    global last_audio_end, last_write_at
    stream = None
    stream_sr = None
    try:  # open the speaker now, not when the first word needs it (that alone can cost ~1s)
        stream = sd.OutputStream(samplerate=voice_sr, channels=1, dtype="float32", latency=AUDIO_LATENCY)
        stream.start()
        stream_sr = voice_sr
    except Exception:
        stream = None
    while True:
        ep, samples, sr, text = play_q.get()
        try:
            if ep == current_epoch():
                if stream is None or stream_sr != sr:
                    if stream is not None:
                        stream.close()
                    stream = sd.OutputStream(
                        samplerate=sr, channels=1, dtype="float32", latency=AUDIO_LATENCY
                    )
                    stream.start()
                    stream_sr = sr

                audio = prepare_audio(samples)
                step = int(sr * 0.05)  # 50 ms writes so a barge-in stops within a blink
                for i in range(0, len(audio), step):
                    if ep != current_epoch():
                        stream.abort()   # throw away audio still buffered
                        stream.start()
                        break
                    stream.write(audio[i:i + step])
                    last_write_at = time.time()
                    if t_audio1 == 0.0:
                        note_first_audio()
                else:
                    if ep == current_epoch():
                        spoken_log.append(text)  # heard in full
        except Exception as e:
            print(f"\n[audio error] {e}", flush=True)
            stream = None  # reopen next time
        finally:
            last_audio_end = time.time()
            play_q.task_done()


# ---------- LLM ----------
def warm_up_llm() -> None:
    # load the model into memory now so your first question isn't slow
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps({
                "model": MODEL,
                "keep_alive": KEEP_ALIVE,
                **({"options": {"num_thread": LLM_THREADS}} if LLM_THREADS else {}),
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=300).read()
    except Exception:
        pass


def stream_reply(messages, on_piece):
    """Streams the reply, calling on_piece for each chunk. Returns (text, finished)."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
    }
    if LLM_THREADS:
        payload["options"] = {"num_thread": LLM_THREADS}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    reply = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            if cancel.is_set():
                return "".join(reply), False
            chunk = json.loads(raw)
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                reply.append(piece)
                on_piece(piece)
            if chunk.get("done"):
                break
    return "".join(reply), True


def model_worker() -> None:
    global last_reply_words, current_question, last_assistant_msg, active_stream
    global reply_t0, t_first_chunk, t_audio1, first_synth_s
    while True:
        item = lines_q.get()
        if item is None:
            return
        text, question = item

        # if you said several things while it was busy, treat it as one thought
        while True:
            try:
                nxt = lines_q.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                return
            text += " " + nxt[0]
            question += " " + nxt[1]

        cancel.clear()
        barge_in.clear()
        spoken_log.clear()
        last_reply_words = set()
        last_reply_seq.clear()
        last_assistant_msg = None
        current_question = question
        reply_t0, t_first_chunk, t_audio1, first_synth_s = time.time(), 0.0, 0.0, 0.0
        my_epoch = current_epoch()
        stream = TextStream(my_epoch)
        active_stream = stream

        with history_lock:
            history.append({"role": "user", "content": text})
            messages = list(history[-MAX_HISTORY:])

        system_text = machine_time_context()

        if SYSTEM_PROMPT:
            system_text = SYSTEM_PROMPT + "\n\n" + system_text

        messages = [{"role": "system", "content": system_text}] + messages

        def on_piece(piece: str) -> None:
            print(piece, end="", flush=True)
            stream.feed(piece)  # the feeder thread speaks it as soon as it's worth speaking

        generating.set()
        print("ai> ", end="", flush=True)
        try:
            reply, finished = stream_reply(messages, on_piece)
            if finished:
                stream.finish()
                msg = {"role": "assistant", "content": reply}
                with history_lock:
                    history.append(msg)
                last_assistant_msg = msg
                print()
            else:
                # cut off mid-reply: only keep what you actually heard in its memory
                spoken = " ".join(spoken_log).strip()
                if spoken:
                    with history_lock:
                        history.append({"role": "assistant", "content": spoken})
                print("  [interrupted]")
        except urllib.error.URLError as e:
            stream.finish()
            with history_lock:
                history.pop()  # don't keep a turn that never got an answer
            print(f"\n[ollama error] {e}  (is `ollama serve` running, is '{MODEL}' in `ollama list`?)")
        except Exception as e:
            stream.finish()
            with history_lock:
                history.pop()
            print(f"\n[error] {e}")
        finally:
            generating.clear()


# ---------- interrupting and resuming ----------
def interrupt(reason: str = "") -> None:
    """You started talking over it: stop speaking, stop generating, remember where it was cut off."""
    global pending_resume
    with interrupt_lock:
        if barge_in.is_set() or not ai_active():
            return
        barge_in.set()
        spoken = " ".join(spoken_log).strip()
        pending_resume = (
            {"question": current_question, "spoken": spoken} if current_question else None
        )
        cancel.set()
        stop_speech()
        if not generating.is_set():
            # it had already finished writing the reply, but you only heard part of it
            with history_lock:
                msg = last_assistant_msg
                if msg is not None:
                    if spoken:
                        msg["content"] = spoken
                    else:
                        history[:] = [m for m in history if m is not msg]
            print(f"  [interrupted{': ' + reason if reason else ''}]", flush=True)
        elif reason:
            print(f"\n  [interrupting: {reason}]", flush=True)


def build_resume_prompt(p) -> str:
    question = p["question"][:300]
    said = p["spoken"][-300:]
    if said:
        return (
            f'Please continue your earlier answer to "{question}" from exactly where you were '
            f'cut off. The last thing I heard you say was: "{said}". Don\'t repeat that or '
            f"start over, just carry on from there."
        )
    return f"Sorry, I cut you off before I heard anything. Please answer this again: {question}"


def resume_now() -> None:
    """Pick the cut-off answer back up (voice: "continue", keyboard: c + Enter)."""
    global pending_resume
    if pending_resume:
        p, pending_resume = pending_resume, None
        lines_q.put((build_resume_prompt(p), p["question"]))
    else:
        print("  [nothing to resume]", flush=True)


def is_stop_cmd(text: str) -> bool:
    """'wait', 'pause', 'wait wait pause pause', 'hold on a second'... but not 'wait, what about X?'"""
    words = tokens(text)
    return (
        bool(words)
        and any(w in STOP_WORDS for w in words)
        and all(w in STOP_WORDS or w in STOP_FILLER for w in words)
    )


class VoiceDetector:
    """Spots you talking over the bot from mic loudness alone (no speech recognition involved)."""

    def __init__(self):
        self.idle_floor = 0.005                       # room noise while it's quiet
        self.history = collections.deque(maxlen=70)   # mic loudness while it was talking (~2 s)

    def reset(self) -> None:
        self.history.clear()

    def update(self, rms: float, playing: bool) -> bool:
        if not playing:
            if rms < self.idle_floor * 4 + 0.002:     # don't learn your own voice as "noise"
                self.idle_floor = 0.97 * self.idle_floor + 0.03 * rms
            self.reset()
            return False

        need = 4 if HEADPHONES else 8                 # ~30 ms frames you must stay loud for
        self.history.append(rms)
        frames = list(self.history)
        recent, older = frames[-need:], frames[:-need]
        if len(recent) < need:
            return False
        if len(older) >= 15:
            base = float(np.percentile(older, 90))    # how loud its own echo normally gets
        elif HEADPHONES:
            base = self.idle_floor
        else:
            return False                              # still learning how loud its echo is
        mean = sum(recent) / need
        return (
            mean > max(VOICE_MIN_RMS, base * VOICE_RATIO)
            and min(recent) > base * 1.3
        )


def voice_monitor() -> None:
    frame_q = queue.SimpleQueue()

    def on_audio(indata, frames, time_info, status):
        frame_q.put(float(np.sqrt(np.mean(indata * indata))))

    try:
        stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", blocksize=480, callback=on_audio
        )
        stream.start()
    except Exception as e:
        print(f"[loudness interrupt is off: couldn't open the mic a second time ({e}). "
              f"Speech and the Enter key still work.]", flush=True)
        return

    det = VoiceDetector()
    while True:
        rms = frame_q.get()
        playing = time.time() - last_write_at < 0.25
        if det.update(rms, playing) and ai_active():
            if SHOW_DEBUG:
                print(f"\n[voice] you're louder than its echo ({rms:.3f}), interrupting", flush=True)
            interrupt("you got louder than its echo")
            det.reset()


def keyboard_worker() -> None:
    """Works even if the recognizer doesn't: Enter = interrupt, c + Enter = continue, or type a message."""
    while True:
        try:
            raw = sys.stdin.readline()
        except Exception:
            return
        if raw == "":
            return  # no keyboard attached
        cmd = raw.strip()
        if not cmd:
            if ai_active():
                interrupt("Enter key")
        elif norm(cmd) in ("c", "continue", "resume", "go on"):
            resume_now()
        else:
            if ai_active():
                interrupt("you typed something")
            print(f"you> {cmd}", flush=True)
            with turn_lock:
                turn_parts.append(cmd)
            arm_turn(0.05)


# ---------- turn-taking ----------
def is_filler_only(text: str) -> bool:
    words = tokens(text)
    return not words or all(w in FILLERS for w in words)


def wait_time(text: str) -> float:
    """How long to wait for you to continue, based on how your last line ended."""
    t = text.strip().lower()
    words = tokens(t)
    if t.endswith(("...", "\u2026", ",", "-")) or (words and words[-1] in TRAIL_WORDS):
        return END_OF_TURN_S + TRAIL_EXTRA_S
    if t.endswith("?"):
        return QUESTION_WAIT_S
    return END_OF_TURN_S


def arm_turn(delay: float) -> None:
    """(Re)start the countdown. Does nothing if you haven't said anything yet."""
    global turn_timer
    with turn_lock:
        if turn_timer:
            turn_timer.cancel()
            turn_timer = None
        if not turn_parts:
            return
        turn_timer = threading.Timer(delay, commit_turn)
        turn_timer.daemon = True
        turn_timer.start()


def commit_turn() -> None:
    """You've been quiet long enough: act on everything you said as one message."""
    global turn_timer, pending_resume
    with turn_lock:
        text = " ".join(turn_parts)
        turn_parts.clear()
        turn_timer = None
    text = re.sub(r"\s+", " ", FILLER_RE.sub("", text)).strip()
    if not text:
        return

    spoken_form = norm(text)
    if is_stop_cmd(text) or HOLD_RE.match(spoken_form):
        # "stop" / "wait": it's already quiet, don't ask the model anything
        if pending_resume:
            print('  [paused - say "continue" to pick it back up]', flush=True)
        return
    if RESUME_RE.match(spoken_form) and pending_resume:
        resume_now()
        return

    # ---------------- wake word gate ----------------
    # Normal speech is ignored unless it starts with "ducky" or "Hey ducky".
    # Saying only "ducky" / "Hey ducky" arms the assistant for a few seconds,
    # so you can pause naturally and then give the command.
    global wake_armed_until

    now = time.time()
    wake_type, command = extract_wake_command(text)

    # "ducky, ..." or "Hey ducky, ..."
    if wake_type == "command":
        wake_armed_until = 0.0
        lines_q.put((command, command))
        return

    # Just "ducky" / "Hey ducky"
    if wake_type == "wake_only":
        wake_armed_until = now + WAKE_TIMEOUT_S
        print('  [ducky listening]', flush=True)
        return

    # We recently heard "ducky", so accept the next line as the command.
    if now < wake_armed_until:
        wake_armed_until = 0.0
        lines_q.put((text, text))
        return

    # Otherwise, ignore ordinary speech.
    if SHOW_DEBUG:
        print(f"  [ignored - no wake word] {text}", flush=True)
    return


# ---------- mic callbacks ----------
def show_partial(text: str) -> None:
    global last_partial, line_open, line_tainted
    if text.strip() and text != last_partial:
        last_partial = text
        if not line_open:
            line_open = True
            line_tainted = ai_active() or time.time() < last_audio_end + 0.7
        if SHOW_DEBUG and ai_active():
            print(f"\n[mic hears] {text}", flush=True)
        if turn_parts:
            arm_turn(HOLD_FALLBACK_S)  # you started talking again: don't answer yet
        if ai_active() and is_user_speech(text):
            interrupt(f'heard "{text[-50:]}"')   # you're cutting in: stop talking right now
    if ai_active():
        return  # don't fight with the reply printing
    print("\r" + text[-term_width():].ljust(term_width()), end="", flush=True)


def handle_line(line) -> None:
    global line_open, line_tainted
    tainted = line_tainted if line_open else (ai_active() or time.time() < last_audio_end + 0.7)
    line_open, line_tainted = False, False
    text = line.text.strip()
    if not ai_active():
        print("\r" + " " * term_width() + "\r", end="")  # clear the partial
    if not text:
        return

    # speaker mode: drop lines that are just its own voice coming back through the mic
    if not HEADPHONES and looks_like_echo(text, tainted):
        if SHOW_DEBUG:
            print(f"\n[dropped as its own echo] {text}", flush=True)
        return

    # "um" / "uh" on its own: you're still thinking, so give it more time
    if is_filler_only(text):
        arm_turn(END_OF_TURN_S + TRAIL_EXTRA_S)
        return

    if ai_active():
        interrupt(f'heard "{text[-50:]}"')  # in case the live partial didn't already catch it
    print(f"you> {text}", flush=True)
    with turn_lock:
        turn_parts.append(text)
    arm_turn(wait_time(text))


def main() -> None:
    global voice_sr, LLM_THREADS
    print("Loading voice...", flush=True)
    engine, voice_sr = load_voice()
    if engine.name == "Piper":
        LLM_THREADS = None  # Piper is light, so the model can use every core

    warm = threading.Thread(target=warm_up_llm, daemon=True)  # load the LLM while the mic loads
    warm.start()

    threading.Thread(target=synth_worker, args=(engine,), daemon=True).start()
    threading.Thread(target=player_worker, daemon=True).start()
    threading.Thread(target=feeder_worker, daemon=True).start()
    threading.Thread(target=model_worker, daemon=True).start()
    threading.Thread(target=keyboard_worker, daemon=True).start()
    if VOICE_INTERRUPT:
        threading.Thread(target=voice_monitor, daemon=True).start()

    mic = (
        MicTranscriber()
        .language("en")
        .on_text(show_partial)
        .on_line(handle_line)
    )
    mic.load()   # blocks on first run while the speech model downloads
    print(f"Loading {MODEL}...", flush=True)
    warm.join(timeout=120)  # so your first question isn't the one that waits for the model
    mic.start()
    mode = "headphones" if HEADPHONES else "speaker mode"
    print(f"Listening... talking to '{MODEL}' ({engine.name} voice, {mode}). "
          f'Talk over it (or press Enter) to interrupt; say "continue" (or type c) to resume. '
          f'Ctrl+C to stop.')

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        with turn_lock:
            if turn_timer:
                turn_timer.cancel()
        cancel.set()
        stop_speech()
        for shutdown in (mic.stop, mic.close):
            try:
                shutdown()
            except BaseException:   # e.g. you pressed Ctrl+C twice
                pass
        lines_q.put(None)
        print("\nStopped.")


if __name__ == "__main__":
    main()
