# NSP to WAV Converter

A Python script for batch converting `.nsp` (IFF) audio files to `.wav` format using ffmpeg, with parallel processing and progress tracking.

This tool is specifically designed for converting audio files from the [Stimmdatenbank](https://stimmdb.coli.uni-saarland.de/) (German Voice Database) to standard WAV format for easier processing and analysis.

**Made available by [SHIP.AI](https://ship-ai.ikim.nrw/)**

## Features

- **Recursive scanning** - Finds all `.nsp` files in a directory tree
- **Parallel processing** - Converts multiple files simultaneously using multiple CPU cores
- **Progress tracking** - Shows a real-time progress bar (when tqdm is installed)
- **Sample rate control** - Optional resampling to a target sample rate
- **Safe defaults** - Won't overwrite existing files unless explicitly told to
- **Dry-run mode** - Preview what will be converted without actually converting

## Requirements

- Python 3.10 or higher
- ffmpeg (must be installed and available in PATH)
- tqdm (optional, for progress bar)

### Installing Dependencies

**ffmpeg:**
- Ubuntu/Debian: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

**tqdm (optional):**
```bash
pip install tqdm
```

## Usage

### Basic Usage

Convert all `.nsp` files in the current directory:
```bash
python convert_nsp_to_wav.py .
```

Convert files in a specific directory:
```bash
python convert_nsp_to_wav.py /path/to/audio/files
```

### Options

| Option | Description |
|--------|-------------|
| `--resample RATE` | Resample audio to specified sample rate (e.g., 44100, 48000) |
| `--overwrite` | Overwrite existing `.wav` files |
| `--dry-run` | Show what would be converted without actually converting |
| `--workers N` | Number of parallel workers (default: CPU count) |

### Examples

**Resample all files to 44.1 kHz:**
```bash
python convert_nsp_to_wav.py . --resample 44100
```

**Use 6 parallel workers:**
```bash
python convert_nsp_to_wav.py . --workers 6
```

**Preview conversion without executing:**
```bash
python convert_nsp_to_wav.py . --dry-run
```

**Overwrite existing WAV files:**
```bash
python convert_nsp_to_wav.py . --overwrite
```

**Combine options:**
```bash
python convert_nsp_to_wav.py /audio/files --resample 48000 --workers 8 --overwrite
```

## Output Format

Converted files are saved with:
- **Format:** WAV
- **Codec:** PCM 16-bit signed little-endian (`pcm_s16le`)
- **Sample rate:** Original rate (unless `--resample` is specified)
- **Location:** Same directory as the source file
- **Filename:** Same name with `.wav` extension

## How It Works

1. Recursively scans the specified directory for `.nsp` files
2. Distributes conversion tasks across multiple worker processes
3. Each worker runs ffmpeg to convert one file at a time
4. Shows progress as conversions complete
5. Reports a summary with conversion statistics

## Exit Behavior

- **Skipped files** - Existing `.wav` files are automatically skipped unless `--overwrite` is used
- **Failed conversions** - Error messages are printed to stderr but don't stop the batch
- **Missing ffmpeg** - Script exits immediately with an error message

## Performance Tips

- The default number of workers equals your CPU core count
- For I/O-bound systems (slow disks), fewer workers may be faster
- For CPU-bound systems (fast SSDs), more workers can help
- Test with `--dry-run` first to see how many files will be processed

## Troubleshooting

**"ffmpeg not found in PATH"**
- Install ffmpeg and ensure it's accessible from the command line

**Progress bar not showing**
- Install tqdm: `pip install tqdm`

**Conversions failing**
- Check ffmpeg error messages in the output
- Verify input files are valid `.nsp` files
- Ensure you have write permissions in the target directory

## License

This script is provided as-is for audio file conversion tasks.
