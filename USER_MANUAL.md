# User Manual: ASD Detection and Support System
## EEG Spectrum-based Autism Spectrum Disorder Detection

---

## Table of Contents
1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Getting Started](#getting-started)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [Data Requirements](#data-requirements)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)
10. [Support & Contact](#support--contact)

---

## Introduction

**Welcome to the ASD Detection and Support System!**

This project leverages EEG (Electroencephalogram) spectrum analysis combined with machine learning to assist in the detection and support of Autism Spectrum Disorder (ASD). This user manual provides comprehensive guidance on how to install, configure, and use the system effectively.

### Key Features
- EEG spectrum analysis and preprocessing
- Machine learning-based ASD detection
- Comprehensive data visualization
- Support tools for clinical and research applications
- Easy-to-use interface for data analysis

### Disclaimer
This system is designed for research and support purposes. It should not be used as a standalone diagnostic tool and must be used in conjunction with professional medical evaluation and diagnosis.

---

## System Overview

### What is ASD Detection?
Autism Spectrum Disorder (ASD) is a developmental disorder characterized by differences in social communication and repetitive behaviors. Early detection can facilitate timely intervention and support.

### How Does This System Work?
The system analyzes EEG signals and their frequency spectrum to identify patterns associated with ASD. It uses machine learning models trained on EEG data to provide detection support.

### Project Components
- **Data Preprocessing**: Cleaning and normalizing EEG data
- **Feature Extraction**: Analyzing frequency spectrum characteristics
- **Machine Learning Model**: Classification system for ASD detection
- **Visualization Tools**: Graphs and charts for data analysis
- **Result Interpretation**: Detailed output reports

---

## Getting Started

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python Version**: Python 3.7 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB available space
- **Dependencies**: See Installation section

### Prerequisites
- Basic understanding of EEG data and ASD
- Familiarity with Jupyter Notebooks
- Python programming knowledge (optional but helpful)

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/ShwetaNagapure/unpuzzel_for_ASD.git
cd unpuzzel_for_ASD
```

### Step 2: Create a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

**Common dependencies include:**
- NumPy - Numerical computing
- Pandas - Data manipulation
- Scikit-learn - Machine learning
- Matplotlib & Seaborn - Data visualization
- MNE - EEG signal processing
- Jupyter - Interactive notebooks

### Step 4: Verify Installation
```bash
jupyter notebook
```
Open a notebook file to ensure all packages are working correctly.

---

## Usage Guide

### Basic Workflow

#### 1. **Launch Jupyter Notebook**
```bash
jupyter notebook
```
This opens the Jupyter interface in your default browser.

#### 2. **Open a Notebook File**
- Navigate to the project directory
- Select a notebook file (`.ipynb`)
- Click to open

#### 3. **Load Your EEG Data**
- Place your EEG data files in the designated folder
- Use the data loading cells in the notebook
- Specify file path and format

#### 4. **Run Analysis**
- Execute cells sequentially (Shift + Enter)
- Monitor progress and outputs
- Review generated visualizations

#### 5. **Interpret Results**
- Review classification results
- Check confidence scores
- Generate report

### Running Different Analysis Types

#### EEG Preprocessing
- Load raw EEG data
- Apply filters and noise removal
- Normalize and standardize data
- Export preprocessed data

#### Spectrum Analysis
- Compute frequency spectrum
- Analyze band-specific features (alpha, beta, theta, delta)
- Generate spectrum visualizations
- Extract statistical features

#### ASD Classification
- Load preprocessed data
- Run machine learning model
- Generate predictions with confidence scores
- Create classification report

### Advanced Features

**Custom Model Training:**
- Prepare labeled dataset
- Split into training/validation sets
- Configure model parameters
- Train custom classifier
- Evaluate performance metrics

**Batch Processing:**
- Process multiple EEG files
- Generate consolidated reports
- Comparative analysis
- Export results

---

## Data Requirements

### EEG Data Format

**Supported Formats:**
- `.edf` (European Data Format)
- `.csv` (Comma-Separated Values)
- `.mat` (MATLAB format)
- `.fif` (MEG/EEG Data Format)

### Data Specifications

| Specification | Requirement |
|---------------|------------|
| **Sampling Rate** | 250 Hz minimum (500 Hz typical) |
| **Duration** | 30 seconds to 5 minutes |
| **Channels** | 1-32 channels supported |
| **File Size** | Up to 100MB per file |
| **Data Type** | Float or Integer values |

### Data Preparation Checklist
- [ ] EEG file in supported format
- [ ] Correct sampling rate documented
- [ ] Channel labels included
- [ ] No major data corruption
- [ ] Subject/session information recorded
- [ ] File naming convention followed

### Naming Convention
```
[SubjectID]_[SessionNum]_[Date]_[Type].edf
Example: ASD001_S01_2026-05-05_baseline.edf
```

---

## Interpreting Results

### Output Report Structure

#### 1. Data Quality Metrics
- Signal-to-noise ratio (SNR)
- Missing data percentage
- Artifact presence

#### 2. Spectral Analysis
- Power distribution across frequencies
- Band-specific features (absolute/relative power)
- Frequency anomalies

#### 3. Classification Results
- **Prediction**: ASD or Control
- **Confidence Score**: 0-100% probability
- **Feature Importance**: Which features contributed most
- **Risk Level**: Low/Medium/High

### Understanding Scores

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| **90-100%** | Very High Confidence | Reliable for clinical consideration |
| **70-89%** | High Confidence | Reliable with caution |
| **50-69%** | Moderate Confidence | Requires additional evaluation |
| **<50%** | Low Confidence | Inconclusive - repeat analysis |

### Report Components
- Executive summary
- Detailed findings
- Statistical analysis
- Visualization charts
- Recommendations
- Limitations and caveats

---

## Troubleshooting

### Common Issues & Solutions

#### Issue: "Module not found" error
**Solution:**
```bash
pip install --upgrade [module_name]
# or reinstall all dependencies
pip install -r requirements.txt
```

#### Issue: Data loading fails
**Verification:**
- Check file format is supported
- Verify file path is correct
- Ensure file is not corrupted
- Try converting to CSV format

#### Issue: Slow performance
**Optimization:**
- Reduce data duration
- Decrease number of channels
- Use GPU acceleration if available
- Close other applications

#### Issue: Inconsistent results
**Steps:**
- Verify data quality
- Check preprocessing parameters
- Ensure consistent sampling rate
- Run multiple times to validate

#### Issue: Jupyter Notebook won't open
**Solution:**
```bash
# Clear cache
jupyter notebook --NotebookApp.ip=0.0.0.0

# Or try specific port
jupyter notebook --port 8889
```

### Getting Help
1. Check the FAQ section below
2. Review notebook comments
3. Check repository issues
4. Contact support (see Support section)

---

## FAQ

### General Questions

**Q: Do I need medical expertise to use this system?**
A: No, but basic understanding of EEG and ASD is helpful. The system is designed for both professionals and researchers.

**Q: How accurate is the detection?**
A: Accuracy depends on data quality and preprocessing. Typical accuracy ranges from 80-95% on validation datasets. Always consult healthcare professionals for diagnosis.

**Q: Can I use EEG data from different devices?**
A: Yes, as long as the format is supported and sampling rate is documented. Standardization may be needed.

**Q: How long does analysis take?**
A: Typically 1-5 minutes per file depending on duration and computer performance.

### Technical Questions

**Q: What Python version should I use?**
A: Python 3.7 or higher. Python 3.9-3.11 recommended.

**Q: Can I run this on Windows/Mac/Linux?**
A: Yes, the system is cross-platform compatible.

**Q: How much storage do I need?**
A: Minimum 2GB for installation. Additional space depends on data volume.

**Q: Can I modify the code?**
A: Yes, the system is open for customization and research purposes.

### Data Questions

**Q: How should I prepare my EEG data?**
A: Ensure proper sampling rate, remove major artifacts, and maintain consistent formatting.

**Q: Can I use data from different subjects together?**
A: Yes, but ensure consistency in data collection protocols.

**Q: What if my EEG has noise?**
A: The system includes preprocessing to filter noise, but high-noise data may reduce accuracy.

**Q: How long should EEG recordings be?**
A: 30 seconds to 5 minutes is optimal. Longer recordings provide more data points.

### Results Questions

**Q: What does "Confidence Score" mean?**
A: It indicates the probability that the classification is correct (0-100%).

**Q: Should I trust a 55% confidence result?**
A: No, results below 70% confidence are inconclusive and should not be used alone for decisions.

**Q: Can I override the system's prediction?**
A: The system provides support analysis, but clinical decisions should involve medical professionals.

---

## Support & Contact

### Documentation
- **Repository**: https://github.com/ShwetaNagapure/unpuzzel_for_ASD
- **README**: See `readme.md` in repository
- **Issues**: Check GitHub Issues for known problems

### Getting Help

1. **GitHub Issues**
   - Search existing issues
   - Create new issue with detailed description
   - Include error messages and data samples

2. **Community Forums**
   - EEG signal processing communities
   - Machine learning discussion boards
   - ASD research forums

3. **Direct Contact**
   - Repository maintainer: ShwetaNagapure
   - Email: [Add contact information]
   - GitHub Profile: https://github.com/ShwetaNagapure

### Reporting Bugs
Include the following when reporting issues:
- Operating system and Python version
- Exact error message
- Steps to reproduce
- Data sample (if possible)
- Expected vs actual behavior

### Feature Requests
- Describe the desired feature
- Explain use case
- Suggest implementation approach
- Submit via GitHub Issues

---

## Additional Resources

### Learning Resources
- [EEG Signal Processing Guide](https://en.wikipedia.org/wiki/Electroencephalography)
- [Autism Spectrum Disorder Information](https://www.autism-society.org/)
- [Machine Learning Basics](https://www.coursera.org/learn/machine-learning)
- [Jupyter Notebook Tutorial](https://jupyter.org/try)

### Related Tools
- MNE-Python: EEG/MEG analysis
- EEGLAB: EEG processing
- BrainVision Analyzer: EEG analysis software
- Scikit-learn: Machine learning library

### Publications & References
- Check repository for papers and citations
- Review project documentation for methodology
- Consult scientific literature on EEG-based ASD detection

---

## Version Information

| Item | Details |
|------|---------|
| **Project Version** | 1.0.0 |
| **Last Updated** | 2026-05-05 |
| **Python Version** | 3.7+ |
| **License** | Check repository |
| **Status** | Active Development |

---

## License & Terms

This project is provided for research and support purposes. Users must comply with:
- Data privacy regulations (GDPR, HIPAA where applicable)
- Ethical guidelines for medical research
- Proper attribution and citation
- Terms specified in project LICENSE file

---

## Changelog

### Version 1.0.0 (2026-05-05)
- Initial release
- Core EEG analysis features
- ASD detection model
- Comprehensive documentation

---

## Appendix: Technical Glossary

| Term | Definition |
|------|-----------|
| **EEG** | Electroencephalogram - recording of brain electrical activity |
| **Spectrum** | Frequency decomposition of EEG signal |
| **Band** | Frequency range (Alpha: 8-12Hz, Beta: 12-30Hz, etc.) |
| **Artifact** | Unwanted noise or interference in EEG |
| **Feature** | Measurable characteristic used for analysis |
| **Classification** | Process of categorizing data into groups |
| **Confidence Score** | Probability measure of prediction accuracy |
| **Preprocessing** | Initial data cleaning and preparation |
| **Sampling Rate** | Number of measurements per second (Hz) |

---

**Thank you for using the ASD Detection and Support System. For questions or feedback, please contact the development team.**

*Disclaimer: This system is a research tool and should not replace professional medical diagnosis and evaluation.*
