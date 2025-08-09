# GOTT - AI Financial Assistant

[still in development]

## Overview

GOTT is an advanced AI model built on Microsoft's Phi-3-mini-4k architecture, designed to assist with financial needs
and decision-making processes. The model can be deployed locally using a quantized 4-bit version (gguf format) and
custom-trained on proprietary financial datasets. This approach ensures data privacy, reduced latency, and tailored
financial insights without cloud dependencies. The system leverages machine learning algorithms to provide intelligent
financial analysis and recommendations based on locally processed CSV data.

## Features

- Financial analysis and forecasting
- Investment recommendations
- Risk assessment
- Portfolio optimization
- Market trend analysis

## System Specifications for Local Deployment

### Minimum Requirements

- **CPU**: x86-64 processor supporting AVX2 instruction set
- **RAM**: 8GB (minimum for 4-bit quantized model)
- **Storage**: 3GB free space (model size ~2.7GB in GGUF format)

### Recommended Specifications

- **CPU**: 6+ cores, AVX2/AVX-512 support
- **RAM**: 16GB
- **Storage**: SSD with 5GB+ free space
- **GPU**: NVIDIA GPU with 4GB+ VRAM

### Software Dependencies

- Python: 3.8+
- Docker (for containerized deployment)
- llama.cpp or similar inference engine (for model execution)
- Microsoft Visual C++ Redistributable 2019+
- CUDA Toolkit 11.8+ (only if using GPU acceleration)

## Contributing

- Fork the repository
- Create your feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Support

For support and queries, please open an issue in the GitHub repository.

## Authors

- Initial work - [prasadm1022]