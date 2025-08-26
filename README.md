# Mandarin-to-Cantonese Neural Machine Translation

This project implements a sequence-to-sequence (Seq2Seq) neural network to translate text from Mandarin Chinese to Cantonese. It leverages the power of the Transformer architecture by fine-tuning a pre-trained BART (Bidirectional and Auto-Regressive Transformer) model on a custom parallel corpus.

The project is structured as a command-line application for training, evaluating, and running the translator, demonstrating strong software engineering and MLOps practices.

---

## Key Features

-   **High-Performance Translation:** Fine-tunes a state-of-the-art BART model for a specialized translation task.
-   **Data Pipeline:** Includes a robust data processing pipeline using PyTorch `Dataset` and `DataLoader` for efficient training on a GPU.
-   **Quantitative Evaluation:** Implements a comprehensive evaluation framework using the **BLEU (Bilingual Evaluation Understudy)** score to measure translation quality.
-   **Modular Code:** The project is structured with clean, reusable functions for data handling, training, and inference.
-   **Command-Line Interface:** Easily train, evaluate, or use the translator directly from the terminal using `argparse`.

---

## Technologies Used

-   **Python**
-   **PyTorch**
-   **Hugging Face Transformers** (for BART model and tokenizer)
-   **NLTK** (for BLEU score evaluation)
-   **NumPy**

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/mandarin-cantonese-nmt.git](https://github.com/your-username/mandarin-cantonese-nmt.git)
    cd mandarin-cantonese-nmt
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data:**
    Place your training and testing `.txt` files (tab-separated Mandarin and Cantonese sentences) inside the `data/` directory.

---

## Usage

The `main.py` script is the entry point for all operations, controlled by the `--mode` argument.

### 1. Training the Model

To fine-tune the BART model on your dataset, run the following command. The script will save the trained model to the specified directory.

```bash
python main.py --mode train --train_file data/training_900.txt --epochs 20 --batch_size 16 --model_save_path ./models/my_translator
