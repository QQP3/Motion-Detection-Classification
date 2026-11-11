# Motion Detection and Classification

A modular, cross-platform computer vision project built with Python and OpenCV for video input handling, motion detection, object tracking, and object classification.

## Overview

Motion Detection and Classification is a computer vision project designed to detect and analyze moving objects in video streams.

The application provides a modular processing pipeline that handles video input, frame preprocessing, background subtraction, motion detection, and object tracking.

The project is designed to support multiple video input sources and target operating systems while maintaining a clear separation between its components.

## Features

- Multiple video input sources.
- Live webcam capture.
- Local video file processing.
- Network video stream support.
- Input frame normalization.
- Image preprocessing.
- Background subtraction.
- Motion mask cleaning.
- Motion detection.
- Object tracking.
- Configurable processing parameters.
- Automated testing.
- Cross-platform development architecture.

Object classification is part of the project's architecture and is an area of ongoing development.

## Supported Operating Systems

The project targets the following operating systems:

- **Linux:** Selected major Linux distributions.
- **Windows:** Windows 10 and Windows 11.
- **macOS:** Supported macOS versions.

Python and OpenCV are used to provide a portable video processing and motion detection pipeline.

Camera availability, video capture capabilities, and supported video backends may vary depending on the operating system, hardware, and OpenCV build.

Each target operating system must be tested independently before being considered officially supported.

## Input Sources

The application is designed to support the following video input sources:

### Webcam

Captures live video from connected cameras.

### Local Video Files

Processes video files stored on the local system.

### Network Video Streams

Processes video from network stream URLs, depending on protocol and backend compatibility.

Input source configuration is maintained in:

`config/input_config.yaml`

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Computer vision and video processing |
| NumPy | Numerical and image processing operations |
| PyYAML | YAML configuration management |

See `requirements.txt` for the project's actual dependencies.

## Project Structure

```text
.
├── config/
│   ├── classification_config.yaml
│   ├── detection_config.yaml
│   └── input_config.yaml
│
├── Datas/
│   └── Links.txt
│
├── docs/
│
├── experiments/
│
├── results/
│
├── src/
│   ├── classification/
│   │   └── __init__.py
│   │
│   ├── detection/
│   │   ├── background_subtractor.py
│   │   ├── mask_cleaner.py
│   │   ├── motion_detector.py
│   │   ├── object_tracker.py
│   │   ├── preprocessing.py
│   │   └── __init__.py
│   │
│   ├── input/
│   │   ├── providers/
│   │   ├── input_manager.py
│   │   ├── input_normalizer.py
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   └── __init__.py
│   │
│   ├── visualization/
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── tests/
│   ├── test_background_subtractor.py
│   ├── test_input_manager.py
│   ├── test_input_normalizer.py
│   ├── test_motion_detector.py
│   ├── test_multi_camera_detection.py
│   ├── test_preprocessing.py
│   ├── test_stream_url_provider.py
│   ├── test_video_file_provider.py
│   ├── test_webcam_manual.py
│   └── test_webcam_provider.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Requirements

Before installing the project, make sure you have:

- Python (a version compatible with the project's dependencies).
- pip.
- Git.
- A webcam or another supported video source if you intend to use live video input.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/QQP3/Motion-Detection-Classification
cd Motion-Detection-Classification
```

### 2. Create a virtual environment

**Linux and macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

The project uses YAML configuration files to separate application settings from implementation logic.

| Configuration file | Purpose |
|---|---|
| `config/input_config.yaml` | Video input configuration |
| `config/detection_config.yaml` | Motion detection configuration |
| `config/classification_config.yaml` | Object classification configuration |

Modify these files to configure the corresponding components.

Available parameters and their accepted values are defined by the actual implementation.

## Running the Application

The main application entry point is:

`main.py`

Run the application using:

```bash
python main.py
```

The application uses the configured input and processing settings.

## Testing

The project includes automated tests for its main input and motion detection components.

Run the test suite using:

```bash
python -m pytest
```

Run an individual test file using:

```bash
python -m pytest tests/test_motion_detector.py
```

For webcam tests requiring physical camera access, ensure that the camera is connected and accessible.

The test suite covers:

- Background subtraction.
- Input management.
- Input normalization.
- Motion detection.
- Multi-camera detection.
- Image preprocessing.
- Network stream input.
- Local video file input.
- Webcam input.

## Cross-Platform Development

The project follows these principles to maintain portability:

- Prefer Python standard-library functionality.
- Use cross-platform APIs whenever possible.
- Avoid hardcoded operating-system-specific paths.
- Keep platform-dependent operations isolated.
- Use configurable input sources.
- Test video backends independently on each target operating system.
- Document platform-specific installation requirements.

Cross-platform compatibility is a development objective and is not guaranteed for every operating system, device, or video backend.

## Development Status

The project is under active development.

Current development areas include:

- Video input management.
- Frame normalization and preprocessing.
- Background subtraction.
- Motion detection.
- Motion mask cleaning.
- Object tracking.
- Classification architecture.

Future development will focus on integrating and improving the complete detection and classification pipeline.

## Contributing

Contributions are welcome.

When contributing:

1. Follow the existing modular project structure.
2. Maintain cross-platform compatibility.
3. Add tests for new functionality.
4. Keep configuration separate from implementation logic.
5. Update documentation when introducing new features.
6. Verify that existing tests continue to pass.

## License

No license has been specified yet.

A license should be added to the repository before the project is distributed for public reuse.
