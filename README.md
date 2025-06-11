# EnglishQAHelper 🤖📚

A lightweight Python tool for **English Question Answering** using BERT with sliding window support for long contexts.

> ⚠️ **Disclaimer:** This project is intended **only as an auxiliary tool** for English reading comprehension practice.  
> Answers generated may be imperfect — please verify with trustworthy sources and do not rely solely on this for exams or academic use.

---

## 🚀 Features

- Uses [`bert-large-uncased-whole-word-masking-finetuned-squad`](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad) model  
- Handles long contexts via sliding window to overcome BERT’s input length limits  
- Outputs **top 2** answer candidates ranked by combined start/end token probabilities  
- Prints detailed start/end token probability distributions for transparency and debugging  
- Provides exact character-level answer positions in the original context  

---

## 🔧 Installation

1. Clone the repository:

```bash
git clone https://github.com/wangyifan349/EnglishQAHelper.git
cd EnglishQAHelper
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the QA script on the example English context/question:

```bash
python qa_with_sliding_window.py
```

Modify the variables `context` and `question` inside `qa_with_sliding_window.py` to test your own texts and queries.

---

## 💡 Why Use This?

- Quickly extract probable answer spans from English passages  
- Understand how sliding window helps BERT deal with long inputs  
- Learn about start/end token probabilities and answer span extraction  
- A useful starter project for building custom QA systems  

---

## ⚠️ Important Notes

- The model and outputs are **not guaranteed 100% accurate**.  
- Results are mainly for **educational and practice purposes**.  
- Always cross-check answers with reliable references.  
- Not suitable for official exam submission or critical decision-making.  

---

## 🙌 Contributing

Feel free to submit issues or pull requests to improve this tool! Suggestions for UI, performance, or additional features are welcome.

---

## 📫 Contact

Created by [Wangyifan]  
GitHub: [https://github.com/wangyifan349](https://github.com/wangyifan349)

---

Happy learning! 🎉📖
