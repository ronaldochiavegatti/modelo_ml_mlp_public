# SVD Scope and Classes

## Selected pathologies (top sample counts in local overview.csv files)

- laringite (Laryngitis): 140 samples
- disfonia_funcional (Funktionelle Dysphonie): 112 samples
- disfonia_psicogenica (Psychogene Dysphonie): 91 samples
- edema_de_reinke (Reinke Edema): 68 samples

Rationale:
- These are the pathologies with the highest sample counts in the available overviews.
- Four classes keeps the baseline balanced and avoids extreme class imbalance.
- The selected groups have relatively stable diagnoses in the SVD notes.

## Inclusion criteria

- Only sustained vowels: /a/, /i/, /u/.
- Only normal pitch recordings.
- One WAV per metadata row.

## Exclusion criteria

- Other recording types (text, numbers, cough, whispered, etc.).
- Non-normal pitch recordings.
- Other pathologies not listed above.

## Metadata schema

Required columns in metadata.csv:
- id: unique recording id; include speaker id as prefix to enable speaker-wise split.
- classe: one of the four class labels above (ASCII, no accents).
- filepath: absolute or relative path to the source WAV.
- sexo: m/f if available.
- idade: integer age if available (can be empty).

Recommended id format:
- <speakerId>_<recordingId>_<vowel>
- Example: 1304_107_a

This keeps the CSV format simple while allowing speaker-based splits.
