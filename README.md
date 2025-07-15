
# 📊 ADT Assistant: MOSFET Data Plotter

---

## 🚀 Project Goal

This project is an interactive assistant designed to help analog designers visualize and analyze **MOSFET data from lookup tabels** using natural language queries.

It is inspired by the **Analog Design Toolkit (ADT)** developed by **MasterMicro**, and is intended to be a helper chat application for analog circuit designers.

The system consists of:
- A synthetic data generator for generating testing data.
- A Streamlit-based chat assistant that can:
  - Parse natural language commands like `"plot gm/Id vs intrinsic_gain by L where Vgs > 0.8"`
  - Create interactive Plotly scatter plots.
  - Apply filtering and evaluate expressions like `gm/gds`, `W/L`, etc.

---

## 🔍 Illustration video:
Tested on **Ubuntu24.04 Machine**:

![alt text](images/ADT.gif)

## 🌟 Main Features

- 📈 **Interactive Plots**: Generate customizable scatter plots using `Plotly`.
- 💬 **Chat Interface**: Use natural language to ask for plots and apply filters.
- 🧪 **Lookup tables** (Not implemented currently): dataset includes parameters like:
  - `Vgs`, `Id`, `gm`, `gds`, `W`, `L`
  - Derived metrics: `gm/Id`, `gm/gds`, `W/L`, `intrinsic_gain`, etc. 
- 🧠 **Expression Handling**: Automatically evaluates derived expressions like `gm/Id`, `intrinsic_gain * 2`, etc.
- 🔍 **Conditional Filtering**: Apply filters like `Id > 1e-5`, `VSB < 0.2`, etc.
- 🧪 **Uses Regular Expressions `re`**: to parse user's input messages

---

## 📦 Requirements

- Python 3.8+

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mosfet-data-plotter.git
cd mosfet-data-plotter
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # For linux, if you are on Windows, use: .\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 💬 Example Commands to Try

In the Streamlit chat box, try typing:

- `plot gmoverid vs intrinsic_gain by L`
- `scatter gm/gds vs Vgs where Vgs > 0.8`
- `plot WoverL vs Id by L where Id > 10u`
- `plot intrinsic_gain * 2 vs Vgs by L where gm > 1e-5`

---

## 🔍 Sample Outputs

| Command | Description |
|--------|-------------|
| `plot gm/gds vs Vgs by L` | Shows how output resistance changes with gate voltage across channel lengths |
| `plot gmoverid vs Id` | Evaluates bias efficiency against current |
| `plot intrinsic_gain vs gm/gds where L > 0.3u` | Gain plotted against gm/gds for longer channel devices |

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit chat assistant for plotting and filtering data |
| `generate_csv.py` | Synthetic MOSFET dataset generator |
| `requirements.txt` | Python package dependencies *(create this manually if not included)* |


## 🔍 Future Work:
- Integrate an LLM to suggest modifications to the Analog IC designer.
- Make the plots more interactive by adding magic cursors.
- Read requirements from the designer (i.e. GBW, Gain, CL) and apply the gm/ID methodology to get an initial design.

---