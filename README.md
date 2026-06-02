# ML Training 🧠

Личный проект по изучению машинного обучения. Включает практические уроки, ноутбуки и скрипты по различным темам ML.

## Дорожная карта

Проект следует роадмапу [MachineLearningRoadmap](https://github.com/justxor/MachineLearningRoadmap) — структурированному плану изучения ML с нуля до продвинутого уровня.

## Структура проекта

```
Lessons/
├── 0. Kangle/              # Введение в ML (Kaggle): модели, валидация, случайные леса
├── 1. Kangle - Pandas/     # Pandas: создание, чтение и запись данных
├── 2. Python. Предобработка данных/  # Предобработка и анализ данных
└── 11. prompt-eng-interactive-tutorial/  # Prompt Engineering (Anthropic, Amazon Bedrock)
```


## Темы

- **Базовое исследование данных** — Pandas, визуализация, понимание данных
- **Модели ML** — линейные модели, деревья решений, случайные леса
- **Валидация и оптимизация** — underfitting / overfitting, кросс-валидация
- **Feature Engineering** — работа с признаками, предобработка
- **Prompt Engineering** — взаимодействие с LLM (Claude, Amazon Bedrock)

## Требования

- Python 3.x
- Библиотеки: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `kaggle`

## Для работы локально с kaggle основные команды

Установка kaggle.
```bash
pip install kaggle
```

### Способ авторизации без браузера каждый раз
1. Зайдите на [kaggle.com/account](https://www.kaggle.com/account)
2. В разделе **API** нажмите **"Create New API Token"**
3. Сохраните полученный токен
4. Поместите файл в `~/.kaggle/access_token` (Linux/Mac) или `C:\Users\<username>\.kaggle\access_token` (Windows)

## Проверка авторизации

```bash
# Проверить, что всё работает
kaggle competitions list
# Или
kaggle datasets list
```

## Полезные команды Kaggle CLI

```bash
# Скачать датасет
kaggle datasets download username/dataset-name

# Скачать соревнование
kaggle competitions download competition-name

# Отправить предсказание
kaggle competitions submit -c competition-name -f submission.csv -m "Message"
```

**Важно:** Без авторизации большинство команд не будут работать. Если ещё не авторизованы — начните с `kaggle auth login`.