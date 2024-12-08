# Repository Analysis Tool with Object-Centric Process Mining

A web-based tool that turns GitHub repository data into rich process mining insights using an object-centric approach. Perfect for understanding development patterns, contributor dynamics, and file lifecycles in your projects.

## 🌟 Key Features

- **Smart Commit Classification**: Automatically categorizes commits using either:
  - Direct parsing of conventional commits
  - ML-based classification for non-standardized commits (76.57% accuracy)
- **Object-Centric Process Mining**: Generates visualizations showing how files, developers, and activities interact
- **Comprehensive Analysis**:
  - Contributor collaboration patterns and clustering
  - File lifecycle analysis
  - Temporal activity patterns
  - Development workflow insights

## 🛠️ Technical Stack

- **Backend**: Flask, Python
- **Data Processing**: pandas, pm4py, pydriller
- **Machine Learning**: DistilBERT, PyTorch
- **Visualization**: matplotlib, seaborn, networkx
- **Process Mining**: PM4Py with OCEL support

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/repository-analysis-tool
cd repository-analysis-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5001`

## 📊 Analysis Outputs

The tool generates several types of insights:

- **Object-Centric Process Models**:
  - Directly-Follows Graphs (OCDFG)
  - Petri Nets (OCPN)
- **Temporal Analysis**:
  - Commit patterns over time
  - Activity heatmaps
- **Contributor Analysis**:
  - Collaboration networks
  - Developer clustering
- **File Analysis**:
  - Lifecycle durations
  - Activity distributions
  - Contributor distributions

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎓 Academic Context

This tool was developed as part of a master's thesis project focusing on bridging repository mining and object-centric process mining. If you use this tool in academic work, please cite:

```bibtex
@mastersthesis{belianina,
  title={Object-Centric Process Mining for Software Repository Analysis},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

## 📞 Contact

For questions or support, please open an issue in the GitHub repository or contact elizabethbelyanina@gmail.com.

## ⚠️ Limitations

- Currently supports only public GitHub repositories
- Large repositories may require significant processing time
- Branch analysis is limited to the main branch
- Bot commits are included in the analysis without distinction
