# Contributing to AI Pneumonia Screening System

First off, thank you for considering contributing to this project! It's people like you that make medical AI tools better and more robust.

## 📜 Code of Conduct
This project strictly adheres to medical AI ethical guidelines. All contributors must ensure that data privacy (HIPAA/GDPR) and model fairness are respected.

## 🛠️ How to Contribute
1. **Fork** the repository on GitHub.
2. **Clone** the project to your own machine.
3. **Commit** changes to your own branch (`git checkout -b feature/AmazingFeature`).
4. **Push** your work back up to your fork (`git push origin feature/AmazingFeature`).
5. Submit a **Pull Request** so that we can review your changes.

## 🧪 Testing and CI/CD
Before submitting a pull request, please ensure your code passes our CI pipeline:
- Run `pytest tests/` locally to ensure no core functionalities are broken.
- Ensure your code follows PEP-8 style guidelines.

## 📊 External Datasets & Bias
If you are contributing new data loaders (e.g., for RSNA or CheXpert), please run `bias_analysis.py` to document demographic parity and ensure no hidden biases are introduced.