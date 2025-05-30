# Tugas Besar II Pembelajaran Mesin

## Overview
A *from scratch* implementation of CNN, RNN, and LSTM machine learning models.

## Setup

To get started with this project, follow these steps:

1.  **Clone the repository.**

2.  **Create a virtual environment:**

    It's highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other Python projects on your system.

    **For Ubuntu/Linux:**

    ```bash
    python3 -m venv venv
    ```

    **For Windows:** 

    ```bash
    python -m venv venv
    ```
    > (note: you may want to just use WSL due to `tensorflow-io-gcs-filesystem==0.37.1`, this dependency isn't supported on Windows and reverting to an older version may require to revert many other libraries)

3.  **Activate the virtual environment:**

    **For Ubuntu/Linux:**

    ```bash
    source venv/bin/activate
    ```

    **For Windows (Command Prompt):**

    ```bash
    venv\Scripts\activate
    ```

    **For Windows (PowerShell):**

    ```powershell
    .\venv\Scripts\Activate.ps1
    ```

4.  **Install dependencies:**

    Once your virtual environment is active, install the project's dependencies (if any) using pip.

    ```bash
    pip install -r requirements.txt
    ```

5.  **Before committing, please syncronize the requirements.txt**
    ```bash
    pip freeze > requirements.txt
    ```

## Running the Program
It is recommended to just refer to the `src/playgrounds/` directory. You can find:
1. `aldy_lstm.ipynb` for the LSTM demo (in the last header section)
2. `gana_rnn.ipynb` for the RNN demo (in the last header section)
3. `kristo_cnn.ipynb` for the CNN demo (in the last header section)

Alternatively, you can refer to the respective model class files in `src/models/` to see the individual methods.

## Author

|Nama|NIM|
|-|-|
|Renaldy Arief Susanto|13522022|
|Kristo Anugrah|13522024|
|Nyoman Ganadipa Narayana|13522066|

## Bonuses Done
1. Batch size, back propagation, and training for CNN