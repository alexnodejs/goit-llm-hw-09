# GoIT · Тема 9 — Агент технічної підтримки на LangGraph v1

Домашнє завдання: агент техпідтримки, що для кожного запиту послідовно **класифікує →
шукає у FAQ → генерує стислу відповідь → визначає ескалацію**.

## Граф (строго послідовний)

```
__start__ → classify_query → search_faq → draft_response → check_escalation → __end__
```

- `classify_query` — тип (`problem`/`question`/`complaint`), категорія, терміновість;
- `search_faq` — 1–2 релевантні пункти FAQ (виконується для **всіх** запитів);
- `draft_response` — відповідь українською (≤50 слів) на основі знайдених пунктів FAQ;
- `check_escalation` — ескалація за правилами (`urgency=high` АБО `type=complaint` АБО
  ключові слова `шахрайство/судов/юридичн`); впливає лише на `needs_escalation` та фінальний текст.

## Результати тестів

| Метрика | Значення | Ціль |
|---|---|---|
| Task Completion Rate | **100% (5/5)** | 100% |
| Average Answer Relevancy | **94%** | ≥ 0.8 |
| Escalation Rate | **40% (2/5)** — запити #2 і #4 | 40% |

## Стек

LangGraph v1 · langchain-openai · модель `openai/gpt-4o-mini` через OpenRouter.

## Запуск

```bash
export OPENROUTER_API_KEY='ваш-ключ-openrouter'
jupyter nbconvert --to notebook --execute --inplace dz_topic_09_Oleksandr_Vasyleiko.ipynb
```

Ключ читається зі змінної середовища `OPENROUTER_API_KEY` — у коді ноутбука його немає.

## Файли

- `dz_topic_09_Oleksandr_Vasyleiko.ipynb` — фінальний ноутбук із виконаним виводом;
- `dz_topic_09_Oleksandr_Vasyleiko.py` — джерело з клітинками (`ipynb-py-convert`).
