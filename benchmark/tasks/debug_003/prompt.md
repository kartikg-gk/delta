A customer noticed that a handful of rows in our CSV export don't match what's in the system. Nothing errors — the file is produced successfully and most rows are fine. The affected rows seem to involve free-text fields.

Find out what's being corrupted and fix it.

Requirements:

- The export must round-trip: re-importing an exported file has to reproduce the original data exactly.
- Handle the awkward cases properly rather than stripping the characters that cause trouble.
- Add a test using data that would have caught this.
- Existing consumers of the export must not break.
