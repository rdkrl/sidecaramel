# Overview calibration fixtures

Raw `Serato Overview` payloads (header + 240 x 16 bytes, no audio) that
Serato DJ Pro wrote for generated test signals. `test_overview_cube.py`
asserts the 6x6x6 colour-cube reading against them.

| fixture | signal | source file |
|---|---|---|
| `amp_sweep_60Hz.blob` | 60 Hz sine, amplitude silent → full | `cmcal_07_amp_sweep_silent_to_full_60Hz.wav` |
| `amp_sweep_1kHz.blob` | 1 kHz sine, amplitude silent → full | `cmcal_08_amp_sweep_silent_to_full_1kHz.wav` |
| `amp_sweep_8kHz.blob` | 8 kHz sine, amplitude silent → full | `cmcal_09_amp_sweep_silent_to_full_8kHz.wav` |
| `mix_60Hz_10kHz.blob` | 60 Hz + 10 kHz | `cmcal_22_mix_60Hz_plus_10kHz.wav` |
| `mix_60Hz_1kHz.blob` | 60 Hz + 1 kHz | `cmcal_24_mix_60Hz_plus_1kHz.wav` |
| `mix_1kHz_10kHz.blob` | 1 kHz + 10 kHz | `cmcal_25_mix_1kHz_plus_10kHz.wav` |
| `mix_60Hz_1kHz_10kHz.blob` | 60 Hz + 1 kHz + 10 kHz | `cmcal_23_mix_60Hz_1kHz_10kHz.wav` |
| `log_sweep_background_0.blob` | log sine sweep; uses background 0 | `DJMesh-Ref-Sweep.wav` |

Extracted 2026-10-05. Each payload was compared against the output of
`sidecaramel.overview_encode` (before and after the cube change) for the
same audio and differs from both, so none of them is an encoder output
written back by this package. Three other calibration files
(`cmcal_04` 1 kHz, `cmcal_06` 10 kHz, `cmcal_19` silence) carried blobs
byte-identical to the encoder output and are deliberately not used.
