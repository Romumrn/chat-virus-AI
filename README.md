# 🦠 Virus Dataset AI Agent

 
The agent answers questions by **querying the dataset** and, when needed, **retrieving information from Wikipedia**.
All results are grounded in explicit sources to avoid unsupported claims.


## Features

* Query viral taxonomy data using natural language
* Count and compare species, genera, and families
* Generate Plotly visualizations (bar charts, pie charts)
* Dataset queries and plots are fully transparent
* External information retrieved only from Wikipedia
* Local LLM execution via Ollama


## Dataset

The app expects a CSV file named:

```
viral_taxo.csv
```

The dataset contains viral taxonomic information (species, genus, family, order, etc.) and associated metadata (genomic, ecological, and collection data) from SRA taxonomy.


## Agent behavior

The assistant follows strict rules:

* No invention of biological facts or counts
* All statements must come from:

  * the dataset, or
  * an explicit Wikipedia tool call
* If information is unavailable, the agent states it clearly
* Dataset analysis and plotting are separated and reproducible


## Tools

* **query_dataframe**: runs pandas code on the dataset and returns a DataFrame
* **plot_dataframe**: creates Plotly figures from the previous query result
* **wikipedia_search**: retrieves biological summaries from Wikipedia
* **...**: ..

## Running the app

### Requirements

* Python 3.9+
* Ollama running locally
* Pandas, Numpy, Ploltly 

### Install dependencies

```bash
pip install streamlit pandas numpy requests plotly
```

### Pull a model

```bash
ollama pull gpt-oss
```

### Start the app

```bash
streamlit run app.py
```

---

## Example questions

* “List virus families with more than 100 species”
* “How many genera are in Retroviridae?”
* “Show a bar chart of species count per family”
* “Compare species counts between two families”

---

## Disclaimer

This tool is for exploratory analysis only.
Results should be verified before scientific or medical use.