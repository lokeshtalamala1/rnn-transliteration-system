# RNN Transliteration System

This project implements **character-level transliteration** using deep learning.  
It is divided into two parts:

- **Part A:** Basic Encoder–Decoder Seq2Seq model (RNN/LSTM/GRU)  
- **Part B:** Encoder–Decoder Seq2Seq model with **Bahdanau Attention**  

The models are trained and evaluated on the **Dakshina dataset** (Google Research).  

---

## 📂 Project Structure

rnn-transliteration-system/
│
├── data/ # Dataset folder (not included in repo)
│ └── dakshina_dataset_v1.0/
│ ├── hi/lexicons/hi.transliteration.train.tsv
│ ├── hi/lexicons/hi.transliteration.dev.tsv
│ └── hi/lexicons/hi.transliteration.test.tsv
│
├── partA/ # Basic Seq2Seq
│ ├── transliteration_partA.ipynb # Notebook with experiments
│ ├── train.py # Script for training & evaluation
│ └── utils.py # Shared utils (vocab, data, models)
│
├── partB/ # Attention-based Seq2Seq
│ ├── transliteration_partB.ipynb # Notebook with experiments + visualizations
│ ├── train.py # Script for training & evaluation
│ └── utils.py # Shared utils (vocab, data, models, attention)
│
├── requirements.txt # Python dependencies
└── README.md # Documentation

---

## 📦 Setup Instructions

### 1. Clone this repo
```bash
git clone <your-repo-url>
cd rnn-transliteration-system
2. Install dependencies
pip install -r requirements.txt
3. Download Dakshina Dataset
Download from: Dakshina dataset
Extract it into:

data/dakshina_dataset_v1.0/
Example Hindi dataset files:

hi.transliteration.train.tsv

hi.transliteration.dev.tsv

hi.transliteration.test.tsv

🚀 Usage
▶️ Part A – Basic Seq2Seq (RNN/LSTM/GRU)
Train with default LSTM:

cd partA
python train.py --data_path ../data/dakshina_dataset_v1.0/hi/lexicons/hi.transliteration.train.tsv
Train with GRU:

python train.py --data_path ../data/dakshina_dataset_v1.0/hi/lexicons/hi.transliteration.train.tsv --cell_type GRU
Change hidden size and iterations:

python train.py --hidden_size 512 --epochs 3000
▶️ Part B – Attention-based Seq2Seq
Train with Attention + LSTM:

cd partB
python train.py --data_path ../data/dakshina_dataset_v1.0/hi/lexicons/hi.transliteration.train.tsv
Train with Attention + GRU:

python train.py --cell_type GRU --epochs 2500
📒 Jupyter Notebooks
Both partA and partB contain Jupyter notebooks:

Part A Notebook (transliteration_partA.ipynb)

Loads Dakshina dataset

Builds vocabularies

Implements Encoder–Decoder Seq2Seq model

Trains and evaluates

Shows sample predictions

Part B Notebook (transliteration_partB.ipynb)

Adds Bahdanau-style attention to decoder

Trains with attention mechanism

Evaluates with attention heatmaps

Shows alignment between input characters and predicted output

Run notebooks with:

jupyter notebook
🔎 Visualizations (Part B)
Attention mechanism improves interpretability by showing which input characters the model is focusing on at each decoding step.

Example Attention Heatmap:

X-axis → input (Latin script characters)

Y-axis → output (Native script characters)

Bright cells → stronger attention weights

This helps debug alignment and shows how attention improves transliteration accuracy.

📊 Results (Indicative)
Part A (Basic Seq2Seq):

Can transliterate short/medium words fairly well

Struggles on long sequences due to fixed-size context vector

Part B (Attention):

Learns better alignment between source and target characters

Handles longer words more accurately

Produces more reliable transliterations

📌 Notes
Default language: Hindi (hi)

To try other Dakshina languages (e.g., Telugu te, Bengali bn), just update --data_path with the respective .tsv.

Both train.py scripts use teacher forcing and allow tuning the ratio with --teacher_forcing.

Models are character-level and trained from scratch.

⚙️ Example Commands Summary
Part A (Basic):
python train.py --data_path ../data/dakshina_dataset_v1.0/hi/lexicons/hi.transliteration.train.tsv --cell_type LSTM --epochs 2000

Part B (Attention):
python train.py --data_path ../data/dakshina_dataset_v1.0/hi/lexicons/hi.transliteration.train.tsv --cell_type GRU --epochs 3000
