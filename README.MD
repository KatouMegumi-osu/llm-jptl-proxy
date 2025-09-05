# LLM Prompt MITM expander / Japanese Translation Enhancer

**A vibe-coded, man-in-the-middle proxy to dramatically improve Japanese-to-English translations from local Large Language Models.**

This project was born from a simple idea and grew organically, feature by feature. It's held together with a love for Japanese VNs and the desire to make LLM translators actually *good*. It's powerful, but don't expect a hardened, enterprise-grade application. Expect a tool that was built to solve a problem, and does it very well.

---

### What is this? The Problem and The Solution

Local LLMs are amazing, but they can struggle with the nuance of Japanese, especially in the context of visual novels, manga, and other creative media. They often miss specific vocabulary, misinterpret complex grammar, confuse names with common nouns, and produce bland or "literally correct but emotionally wrong" translations.

This script acts as a **man-in-the-middle proxy** that sits between your application and your local LLM (like KoboldCpp). It intercepts your translation prompt, intelligently enriches it with a massive amount of context, and then sends the enhanced prompt to the LLM.

It gives the LLM a comprehensive "cheat sheet" for the specific sentence it's translating, forcing it to be more accurate.

### How It Works

The process is simple but incredibly effective:

`[Your App] ---> [This Proxy] ---> [Your Local LLM]`

1.  **Intercept:** The proxy catches the API request from your translation app.
2.  **Analyze:** It performs a deep analysis on the Japanese text in your prompt:
    *   **Vocabulary:** Identifies words and looks them up in a massive dictionary (JMdict).
    *   **Named Entities (NER):** Recognizes and labels character names, locations, etc.
    *   **Grammar:** Performs a full syntactic dependency parse to understand the sentence structure.
3.  **Append:** It attaches all this useful information to the end of your prompt in three clean appendices.
4.  **Forward:** It sends the new, super-charged prompt to your LLM.
5.  **Return:** It streams the LLM's high-quality translation back to your app, which is none the wiser.

### Key Features

*   **Comprehensive Lexical Analysis:** Uses a nearly 500,000-entry Japanese-English dictionary (`JMdict`) to provide definitions for words in the prompt.
*   **Deep Grammatical Analysis:** Uses the powerful **GiNZA/spaCy** model to perform a dependency parse, breaking down the sentence's grammatical structure for the LLM.
*   **Named Entity Recognition (NER):** Identifies and labels character names (`PERSON`), locations, etc., helping the LLM with context and gendering.
*   **Hot-Reloading Custom Dictionary:** The real magic. Use a `custom_dict.json` to override default translations, add slang, or define character-specific vocabulary, and see the changes **instantly** without restarting the server.
*   **Asynchronous & Threaded:** Built with FastAPI, the proxy is non-blocking and runs heavy NLP tasks in separate threads, keeping it responsive.

### The Power of `custom_dict.json`

This file is your personal control panel for the translator's brain. It's a simple JSON file where you can define your own vocabulary. It is **checked on every single request**, meaning you can edit it while the proxy is running.

**It has two primary powers:**

1.  **Priority Overrides:** Any definition in `custom_dict.json` will be used *instead* of the main dictionary's definition. This lets you simplify long definitions or enforce a specific translation for a word.
2.  **Add New Words:** You can add words that don't exist in the main dictionary, like character nicknames, modern slang, or specialized onomatopoeia.

**Example `custom_dict.json`:**
```json
{
    "がんばる": "to try one's absolute best (ganbaru)",
    "かいちょ": "Kaicho (President/Captain, a specific nickname)",
    "マジ": "for real?! (slang)",
    "れろれろ": "Invasive, skillful licking.",
    "reru": "Invasive, deep sucking/slurping sound."
}
```

---

### Setup and Installation

Follow these steps carefully to get the proxy running.

#### 1. Prerequisites
*   You need **Python 3.9+** installed.
*   You have a local LLM server running that exposes an OpenAI-compatible API endpoint (e.g., KoboldCpp).

#### 2. Download the Code
Clone this repository or download the files (`run.py`, `parse_jmdict.py`, `requirements.txt`) into a new folder.

#### 3. Install Python Dependencies
Open your terminal in the project folder and run:
```sh
pip install -r requirements.txt
```

#### 4. Set up the JMdict Dictionary (unnecessary unless you want to update it)
This is a one-time setup process to create your local dictionary file.
a. Go to the [JMdict Project Page](http://www.edrdg.org/jmdict/j_jmdict.html).
b. Download the **`JMdict_e.gz`** file.
c. **Unzip it.** You will get a large XML file named `JMdict_e`.
d. Place the `JMdict_e` file **in the same folder** as the Python scripts.
e. Run the parser script from your terminal:
```sh
python parse_jmdict.py
```
This might take a minute or two and generate a `jmdict.json` file. You can now delete the `JMdict_e` file if you wish.

#### 6. Create Your Custom Dictionary (Optional but Recommended)
Create an empty file named `custom_dict.json` in the same folder. You can add your custom definitions to it now or later while the proxy is running.

---

### Running the Proxy

1.  Open your terminal in the project folder.
2.  Run the following command to start the server:
    ```sh
    uvicorn run:app --host 0.0.0.0 --port 8001
    ```
    *(You can change the `--port` to whatever you like, as long as it doesn't conflict with your LLM server.)*

3.  You should see output confirming that the tokenizer, dictionary, and GiNZA model have loaded successfully.

### Usage

The final step is to tell your application to talk to the proxy instead of directly to your LLM.

In your client application's settings, change the API endpoint URL:

*   **BEFORE:** `http://localhost:5001/v1/chat/completions`
*   **AFTER:** `http://localhost:8001/v1/chat/completions`

That's it! Your application will now send its prompts through the proxy, and you'll start receiving enhanced translations.

Enjoy your new high-fidelity translation experience!