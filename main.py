from typing import Iterable
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from pathlib import Path
import srt
from tqdm import tqdm

DEFAULT_SRT_FILE_FOLDER = Path("./SRT_File")


class Handler:
    movie_file_path: Path
    srt_file_path: Path
    movie_transcribe_content: Iterable[Segment]

    def __init__(self, movie_file_path: Path) -> None:
        try:
            if not DEFAULT_SRT_FILE_FOLDER.exists():
                DEFAULT_SRT_FILE_FOLDER.mkdir(parents=True, exist_ok=True)
            self.movie_file_path = movie_file_path
            self.srt_file_path = DEFAULT_SRT_FILE_FOLDER.joinpath(
                movie_file_path.stem
            ).with_suffix(".srt")

            if not self.srt_file_path.exists():
                self.srt_file_path.touch()
        except Exception as e:
            print(f"Error initializing Handler for '{movie_file_path}': {e}")
            raise

    def transcription(self) -> None:
        try:
            file_path = str(self.movie_file_path)
            model = "./module/faster-whisper-large-v3"
            model = WhisperModel(model, device="cpu", compute_type="int8",cpu_threads=12,num_workers=1)
            segments, info = model.transcribe(
                file_path,
                beam_size=10,
                best_of=10,
                temperature=0,
                initial_prompt="以下是教学课程视频的录音,请注意识别专有名词和技术术语:",
                vad_filter=True,
                word_timestamps=True,
                condition_on_previous_text=False,
            )

            print(
                "Detected language '%s' with probability %f"
                % (info.language, info.language_probability)
            )

            self.movie_transcribe_content = segments
        except Exception as e:
            print(f"Transcription error for '{self.movie_file_path}': {e}")
            self.movie_transcribe_content = []

    def save_result(self) -> None:
        try:
            with open(self.srt_file_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(tqdm(self.movie_transcribe_content), start=1):
                    start = self._format_timestamp(segment.start)
                    end = self._format_timestamp(segment.end)
                    f.write(f"{i}\n{start} --> {end}\n{segment.text.strip()}\n\n")
            print(f"Result saved to '{self.srt_file_path}'")
        except Exception as e:
            print(f"Error saving result for '{self.movie_file_path}': {e}")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def get_handler(folder_path: Path) -> list[Handler]:
    handlers = []
    if folder_path.is_dir():
        for file in folder_path.iterdir():
            if file.is_file():
                try:
                    handlers.append(Handler(file))
                except Exception as e:
                    print(f"Skipping file '{file.name}' due to error: {e}")
        return handlers
    else:
        print(f"Directory not found: {folder_path}")
        return []


def main():
    try:
        file = Path("./testfile/course.mp4")
        # 扫描该文件所在的文件夹
        handlers = get_handler(file.parent)
        for hd in handlers:
            print(f"Processing: {hd.movie_file_path.name}")
            hd.transcription()
            hd.save_result()
    except Exception as e:
        print(f"Main execution error: {e}")


if __name__ == "__main__":
    main()
