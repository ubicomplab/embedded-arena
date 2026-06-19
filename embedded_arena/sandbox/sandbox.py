import os
import shutil
import subprocess
from pathlib import Path
import fcntl
import hashlib
import requests
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()


def env_value(name: str, default: str | Path) -> str:
    """Read an environment variable, treating blank .env entries as unset."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return str(default)
    return value


CACHE_DIR = Path(
    env_value("EMBEDDED_ARENA_CACHE_DIR", Path.home() / ".cache" / "embedded-arena")
).expanduser()
DEFAULT_SANDBOX_PATH = env_value(
    "EMBEDDED_ARENA_SANDBOX_PATH",
    CACHE_DIR / "sandboxes" / "default",
)
DOCKERFILE_DIR = Path(env_value("EMBEDDED_ARENA_DOCKER_DIR", CACHE_DIR / "docker")).expanduser()
DOCKER_CONTAINER_TAG = env_value("EMBEDDED_ARENA_DOCKER_TAG", "embedded-arena:sandbox")
SANDBOX_MARKER = ".embedded-arena-sandbox"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_DIR.parent
MAX_COMMAND_PARTS = 256
MAX_COMMAND_PART_LENGTH = 8192

# To avoid getting blocked by websites, we use a user agent string.
USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "priority": "u=0, i",
    "sec-ch-ua": '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
    "sec-ch-ua-arch": '"arm"',
    "sec-ch-ua-bitness": '"64"',
    "sec-ch-ua-full-version-list": '"Google Chrome";v="141.0.7390.123", "Not?A_Brand";v="8.0.0.0", "Chromium";v="141.0.7390.123"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-model": '""',
    "sec-ch-ua-platform": '"macOS"',
    "sec-ch-ua-platform-version": '"15.6.1"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
}


class Sandbox:
    """
    Isolates a working director for agent use (download files from the internet, run arbitrary commands, and read files)
    Requires Docker. By default the sandbox image includes Python 3.11, git, and bash.
    """

    def __init__(
        self,
        sandbox_path=DEFAULT_SANDBOX_PATH,
        base_image="python:3.11-slim-bookworm",
        install=(
            "RUN apt-get update && apt-get install -y --no-install-recommends "
            "git bash && rm -rf /var/lib/apt/lists/*\n"
            "RUN python -m pip install --no-cache-dir numpy pyyaml pillow pyarrow\n"
            "RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch"
        ),
        hide_build_output=True,
        network_access=False,
    ):
        """Instantiates sandbox_path as the working directory for the agent"""

        self.sandbox_path = str(Path(sandbox_path).resolve())
        self.network_access = network_access
        dockerfile_contents = f"FROM {base_image}\n{install}\nWORKDIR /app"
        dockerfile_hash = hashlib.sha256(dockerfile_contents.encode()).hexdigest()[:12]
        self.container_tag = f"{DOCKER_CONTAINER_TAG}-{dockerfile_hash}"

        self._ensure_safe_sandbox_root()
        os.makedirs(self.sandbox_path, exist_ok=True)
        marker = Path(self.sandbox_path) / SANDBOX_MARKER
        marker.write_text("EmbeddedArena sandbox directory. Contents may be deleted.\n", encoding="utf-8")
        DOCKERFILE_DIR.mkdir(parents=True, exist_ok=True)

        lock_path = DOCKERFILE_DIR / f"sandbox-{dockerfile_hash}.Dockerfile.lock"
        try:
            with lock_path.open("w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                image_exists = subprocess.run(
                    ["docker", "image", "inspect", self.container_tag],
                    check=False,
                    capture_output=True,
                    timeout=30,
                ).returncode == 0
                if not image_exists:
                    dockerfile_path = DOCKERFILE_DIR / f"sandbox-{dockerfile_hash}.Dockerfile"
                    dockerfile_path.write_text(dockerfile_contents, encoding="utf-8")

                    subprocess.run(
                        [
                            "docker",
                            "build",
                            "-f",
                            str(dockerfile_path),
                            str(DOCKERFILE_DIR),
                            "-t",
                            self.container_tag,
                        ],
                        check=True,
                        capture_output=hide_build_output,
                        timeout=1800,
                    )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Docker did not respond while preparing the sandbox image. "
                "Restart Docker Desktop or verify `docker image inspect` works, then rerun."
            ) from exc

    def _ensure_safe_sandbox_root(self) -> None:
        """Refuse sandbox paths that could erase source code or user data."""
        path = Path(self.sandbox_path).expanduser().resolve()
        project_root = PROJECT_ROOT.resolve()
        package_dir = PACKAGE_DIR.resolve()
        home = Path.home().resolve()
        forbidden_exact = {Path("/").resolve(), home, project_root, package_dir}
        if path in forbidden_exact:
            raise RuntimeError(f"Refusing unsafe sandbox path: {path}")
        try:
            path.relative_to(project_root)
        except ValueError:
            pass
        else:
            raise RuntimeError(
                f"Refusing sandbox path inside the EmbeddedArena repository: {path}. "
                "Use the default cache path or set EMBEDDED_ARENA_SANDBOX_PATH outside the repo."
            )
        if path.exists():
            if (path / ".git").exists() or (path / "pyproject.toml").exists():
                raise RuntimeError(f"Refusing sandbox path that looks like a project root: {path}")
            marker = path / SANDBOX_MARKER
            contents = [child for child in path.iterdir() if child.name != ".DS_Store"]
            if contents and not marker.exists():
                raise RuntimeError(
                    f"Refusing to clean non-empty unmarked sandbox path: {path}. "
                    f"Create an empty directory or choose another --sandbox-path. "
                    f"EmbeddedArena only cleans directories marked with {SANDBOX_MARKER}."
                )

    def clean(self):
        """Clear sandbox contents without deleting the sandbox directory itself."""

        self._ensure_safe_sandbox_root()
        root = Path(self.sandbox_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        marker = root / SANDBOX_MARKER
        marker.write_text("EmbeddedArena sandbox directory. Contents may be deleted.\n", encoding="utf-8")
        for child in root.iterdir():
            if child.name == SANDBOX_MARKER:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _resolve_relative_path(self, relative_path: str):
        path = (Path(self.sandbox_path) / relative_path).resolve()
        try:
            path.relative_to(self.sandbox_path)
        except ValueError as exc:
            raise ValueError("must not access file outside sandbox") from exc
        return str(path)

    def open(self, relative_path: str, mode: str):
        """Returns an open file handle for any path inside sandbox_path"""

        path = self._resolve_relative_path(relative_path)
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return open(path, mode)

    def _validate_command(self, cmd: list[str]) -> None:
        if not isinstance(cmd, list) or not cmd:
            raise ValueError("cmd must be a non-empty list of strings")
        if len(cmd) > MAX_COMMAND_PARTS:
            raise ValueError(f"cmd may contain at most {MAX_COMMAND_PARTS} parts")
        for part in cmd:
            if not isinstance(part, str):
                raise ValueError("cmd must contain only strings")
            if "\x00" in part:
                raise ValueError("cmd entries must not contain NUL bytes")
            if len(part) > MAX_COMMAND_PART_LENGTH:
                raise ValueError(
                    f"cmd entries may contain at most {MAX_COMMAND_PART_LENGTH} characters"
                )

    def run(self, cmd: list[str], timeout_seconds=120):
        """Runs an arbitrary command with sandbox_path as the working directory."""

        self._validate_command(cmd)
        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--workdir",
            "/app",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--memory",
            "2g",
            "--cpus",
            "2",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=256m",
            "--mount",
            f"type=bind,source={self.sandbox_path},target=/app",
        ]
        if not self.network_access:
            docker_cmd.extend(["--network", "none"])
        docker_cmd.append(self.container_tag)
        docker_cmd.extend(cmd)
        res = subprocess.run(
            docker_cmd,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return res.stdout.decode(), res.stderr.decode(), res.returncode

    def download_url_as_markdown(self, url, destination_path, timeout_seconds=120):
        """Scrapes a website url and downloads it as markdown"""

        if not self.network_access:
            raise PermissionError("network access is disabled for this sandbox")

        destination_path = self._resolve_relative_path(destination_path)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        response = requests.get(url, timeout=timeout_seconds, headers=USER_AGENT)
        response.raise_for_status()

        if ".pdf" in url.lower() or response.headers.get(
            "content-type", ""
        ).lower().startswith("application/pdf"):
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered

            converter = PdfConverter(artifact_dict=create_model_dict())
            rendered = converter(BytesIO(response.content))
            text, _, _ = text_from_rendered(rendered)
        else:
            import html_to_markdown

            text = (
                html_to_markdown.convert(
                    response.content.decode("utf-8", errors="ignore")
                ).content
                or ""
            )

        with open(destination_path, "w") as f:
            f.write(text)

        return True

    def search_google(self, query: str, num_results: int = 5) -> str:
        """Use the Google search engine to find urls matching a query (with a title and relevant snippet)"""

        if not self.network_access:
            raise PermissionError("network access is disabled for this sandbox")

        search_url = (
            f"https://www.googleapis.com/customsearch/v1"
            f"?q={query}"
            f"&key={os.environ['GOOGLE_SEARCH_API_KEY']}"
            f"&cx={os.environ['GOOGLE_SEARCH_ENGINE_ID']}"
            f"&num={num_results}"
        )
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("items", []):
            results.append(
                f"* {item.get('title')} ({item.get('link')}): {item.get('snippet')}"
            )
        return "\n".join(results) if results else "No relevant results found."

    def copy_to_playground(self, src_path: str, dest_path: str):
        """Copies the given src_path to the dest_path. src_path can be a directory. The parent directory of dest_path must already exist"""

        dest_path = self._resolve_relative_path(dest_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if os.path.isdir(src_path):
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)

            def _ignore_build(_dir: str, names: list[str]) -> list[str]:
                # ESP-IDF (and similar) CMakeCache.txt embeds absolute paths; copying
                # `build/` breaks builds after the project is relocated into the sandbox.
                return [n for n in names if n == "build"]

            shutil.copytree(src_path, dest_path, ignore=_ignore_build)
        else:
            shutil.copy(src_path, dest_path)


if __name__ == "__main__":
    # Example usage
    playground = Sandbox()
    playground.clean()

    with playground.open("test.py", "w") as f:
        f.write("print('hello world!')")

    out, err, returncode = playground.run(["python", "test.py"])
    print("OUTPUT:")
    print(out)
    if returncode != 0:
        print(f"ERROR (exitcode={returncode}):")
        print(err)

    # playground.download_url_as_markdown("https://koellabs.com", "koellabs.md")

    # print(playground.search_google("MAX78000 datasheet"))
    # playground.download_url_as_markdown(
    #     "https://www.analog.com/media/en/technical-documentation/data-sheets/MAX78000.pdf",
    #     "MAX78000_datasheet.md",
    # )
