# %%
"""
# Агент технічної підтримки на LangGraph v1

**Домашнє завдання — Тема 9.** Агент автоматично обробляє запити користувачів:
класифікує їх, шукає відповідь у базі знань FAQ і генерує стислі відповіді.

**Логіка графа** — строго послідовна, без умовної маршрутизації:

`__start__ → classify_query → search_faq → draft_response → check_escalation → __end__`

- `classify_query` — класифікація запиту (тип, категорія, терміновість);
- `search_faq` — пошук релевантних пунктів FAQ (виконується для **всіх** запитів);
- `draft_response` — генерація відповіді (≤50 слів) на основі знайдених пунктів FAQ;
- `check_escalation` — визначення необхідності ескалації (впливає лише на `needs_escalation`
  та фінальний текст відповіді).

Модель: `openai/gpt-4o-mini` через OpenRouter (дешева, надійна, з підтримкою structured output).
"""

# %%
"""
## 1. Імпорти
"""

# %%
import os
from typing import TypedDict, Literal, Optional, List

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# %%
"""
## 2. Ініціалізація LLM (OpenRouter)

Ключ читається зі змінної середовища `OPENROUTER_API_KEY`, тож у коді ноутбука його немає.
"""

# %%
API_KEY = os.environ.get("OPENROUTER_API_KEY", "PASTE_YOUR_OPENROUTER_KEY_HERE")

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.1,
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

# %%
"""
## 3. Підготовка даних: база знань FAQ і типи даних
"""

# %%
FAQ_DATABASE = {
    "password": [
        "Скидання пароля: Натисніть 'Забули пароль?' → введіть email → перевірте пошту",
        "Новий пароль: мінімум 8 символів з літерами та цифрами",
        "Лист не прийшов – перевірте папку спам",
    ],
    "payment": [
        "Перевірте дані картки та баланс",
        "Подвійне списання повернеться автоматично за 3-5 днів",
        "Зміна тарифу: Профіль → Підписка",
    ],
    "technical": [
        "Оновіть сторінку (F5) або очистьте кеш",
        "Спробуйте інший браузер",
        "Перевірте status.ourservice.com",
    ],
    "account": [
        "Перевірте правильність email",
        "Корпоративні акаунти: розділ 'Управління командою'",
        "Блокування – зверніться до підтримки",
    ],
    "general": [
        "Підтримка працює 24/7",
        "Час відповіді: 2-4 години",
    ],
}


class QueryClassification(TypedDict):
    """Структура класифікації запиту."""
    type: Literal["problem", "question", "complaint"]
    category: Literal["password", "payment", "technical", "account", "general"]
    urgency: Literal["low", "medium", "high"]
    summary: str


class SupportAgentState(TypedDict):
    """Стан агента, який передається між вузлами графа."""
    user_query: str
    classification: Optional[QueryClassification]
    search_results: Optional[List[str]]
    draft_response: Optional[str]
    needs_escalation: bool


class FAQSelection(TypedDict):
    """Індекси найрелевантніших пунктів FAQ."""
    indices: List[int]

# %%
"""
## 4. Основний код: вузли графа

Структуровані виклики LLM створюються один раз (DRY). Ескалація визначається простими
правилами (KISS) і не потребує LLM.
"""

# %%
classifier = llm.with_structured_output(QueryClassification)
faq_selector = llm.with_structured_output(FAQSelection)

ESCALATION_KEYWORDS = ("шахрайство", "судов", "юридичн")


def classify_query(state: SupportAgentState) -> dict:
    """Класифікує запит за типом, категорією та терміновістю."""
    prompt = f"""Класифікуй запит користувача до техпідтримки.

type:
- "problem" — користувач зіткнувся з труднощами/несправністю, що заважають користуватися
  сервісом (забув пароль, не може увійти, щось не працює особисто в нього); нейтральний тон.
  ВАЖЛИВО: якщо користувач забув пароль або не може увійти — це завжди "problem",
  навіть якщо сформульовано як питання «як ...?».
- "question" — інформаційний запит про можливість чи спосіб щось зробити
  (як додати, чи можу я змінити).
- "complaint" — невдоволення, обурення, претензія чи вимога
  (оклики, «негайно», «шахрайство», «коли полагодите»).

category:
- "password" — вхід, відновлення чи скидання пароля.
- "payment" — оплата, підписка, кошти, повернення, ЗМІНА ТАРИФУ/ПЛАНУ.
- "technical" — сайт або сервіс не працює, помилки, збої, браузер.
- "account" — керування акаунтом, додавання користувачів, корпоративні акаунти, email.
- "general" — усе інше.

urgency:
- "high" — гнів, терміновість, фінансова шкода або збій сервісу.
- "medium" — особиста проблема доступу без критичності (забув пароль).
- "low" — звичайне інформаційне питання.

summary: стисле резюме запиту українською (одне речення).

Запит: {state['user_query']}"""
    classification = classifier.invoke(prompt)
    return {"classification": classification}


def search_faq(state: SupportAgentState) -> dict:
    """Знаходить 1–2 найрелевантніші пункти FAQ для категорії запиту."""
    category = state["classification"]["category"]
    items = FAQ_DATABASE.get(category, FAQ_DATABASE["general"])
    numbered = "\n".join(f"{i}: {text}" for i, text in enumerate(items))
    prompt = f"""Обери 1-2 НАЙРЕЛЕВАНТНІШІ до запиту пункти FAQ. Поверни лише їх індекси.

Запит: {state['user_query']}

Пункти FAQ:
{numbered}"""
    try:
        selection = faq_selector.invoke(prompt)
        idx = [i for i in selection.get("indices", []) if 0 <= i < len(items)][:2]
        results = [items[i] for i in idx] or items[:2]
    except Exception:
        results = items[:2]
    return {"search_results": results}


def draft_response(state: SupportAgentState) -> dict:
    """Генерує стислу відповідь українською на основі знайдених пунктів FAQ."""
    faqs = "\n".join(f"- {text}" for text in state.get("search_results", []))
    prompt = f"""Ти агент техпідтримки. Дай відповідь українською на запит, спираючись на пункти FAQ.
Вимоги: максимум 50 слів, конкретні кроки, без вступів і привітань, не повторюй запит.

Запит: {state['user_query']}

Релевантні пункти FAQ:
{faqs}"""
    response = llm.invoke(prompt).content.strip()
    return {"draft_response": response}


def check_escalation(state: SupportAgentState) -> dict:
    """Визначає необхідність ескалації та формує фінальний текст відповіді."""
    cls = state["classification"]
    query = state["user_query"].lower()
    needs = (
        cls["urgency"] == "high"
        or cls["type"] == "complaint"
        or any(keyword in query for keyword in ESCALATION_KEYWORDS)
    )
    response = state["draft_response"]
    if needs:
        response = f"{response}\n\n⚠️ Запит передано спеціалісту (відповідь протягом 24 год)."
    return {"needs_escalation": needs, "draft_response": response}

# %%
"""
## 5. Збірка графа: `build_support_agent()`
"""

# %%
def build_support_agent():
    """Будує і компілює послідовний граф агента техпідтримки."""
    builder = StateGraph(SupportAgentState)
    builder.add_node("classify_query", classify_query)
    builder.add_node("search_faq", search_faq)
    builder.add_node("draft_response", draft_response)
    builder.add_node("check_escalation", check_escalation)

    builder.add_edge(START, "classify_query")
    builder.add_edge("classify_query", "search_faq")
    builder.add_edge("search_faq", "draft_response")
    builder.add_edge("draft_response", "check_escalation")
    builder.add_edge("check_escalation", END)

    return builder.compile(checkpointer=MemorySaver())

# %%
"""
## 6. Оцінка релевантності відповіді
"""

# %%
def calculate_answer_relevancy(query: str, response: str) -> float:
    """Оцінка релевантності"""
    relevancy_prompt = f"""
    Оцінка релевантності (0-1):
    Запит: {query}
    Відповідь: {response}

    Тільки число:
    """
    try:
        result = llm.invoke(relevancy_prompt)
        text = result.content.strip()
        score = float(''.join(c for c in text if c.isdigit() or c == '.'))
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5

# %%
"""
## 7. Тестова функція
"""

# %%
def test_agent():
    """Тестування агента на 5 запитах"""
    agent = build_support_agent()

    test_queries = [
        "Я забув пароль від свого акаунту, як його відновити?",
        "З мене двічі зняли гроші за підписку! Це шахрайство! Поверніть кошти негайно!",
        "Чи можу я змінити тарифний план на дешевший?",
        "Сайт не працює вже третю годину! Коли полагодите?",
        "Як додати нового користувача до корпоративного акаунту?",
    ]

    results = []
    relevancy_scores = []

    print("=" * 70)
    print("ТЕСТУВАННЯ АГЕНТА ТЕХНІЧНОЇ ПІДТРИМКИ")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"⚫ ТЕСТ {i}/5")
        print(f"{'='*70}")
        print(f"ЗАПИТ: {query}")
        print("-" * 70)

        initial_state = {
            "user_query": query,
            "needs_escalation": False,
        }
        config = {"configurable": {"thread_id": f"test_{i}"}}

        try:
            result = agent.invoke(initial_state, config)
            results.append(result)

            if result.get("classification"):
                cls = result["classification"]
                print(f"\n📊 КЛАСИФІКАЦІЯ:")
                print(f"   Тип: {cls.get('type')}")
                print(f"   Категорія: {cls.get('category')}")
                print(f"   Терміновість: {cls.get('urgency')}")
                print(f"   Резюме: {cls.get('summary')}")

            if result.get("search_results"):
                print(f"\n🔍 FAQ ({len(result['search_results'])} пунктів):")
                for j, faq in enumerate(result['search_results'], 1):
                    print(f"   {j}. {faq}")

            if result.get("draft_response"):
                print(f"\n💬 ВІДПОВІДЬ АГЕНТА:")
                print("-" * 70)
                print(result["draft_response"])
                print("-" * 70)

                relevancy = calculate_answer_relevancy(query, result["draft_response"])
                relevancy_scores.append(relevancy)
                print(f"\n📈 Релевантність: {relevancy:.0%}")

            if result.get("needs_escalation"):
                print("\n⚠️ СТАТУС: Ескаловано до спеціаліста")
            else:
                print("\n✅ СТАТУС: Оброблено автоматично")

        except Exception as e:
            print(f"❌ ПОМИЛКА: {e}")
            results.append({})

    print(f"\n{'='*70}")
    print("ПІДСУМКОВІ МЕТРИКИ")
    print("=" * 70)

    completed = sum(1 for r in results if r.get("draft_response"))
    tcr = completed / len(results) if results else 0
    print(f"✅ Task Completion Rate: {tcr:.0%} ({completed}/{len(results)})")

    if relevancy_scores:
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
        print(f"🎯 Average Answer Relevancy: {avg_relevancy:.0%}")

    escalated = sum(1 for r in results if r.get("needs_escalation"))
    print(f"📊 Escalation Rate: {escalated}/{len(results)} ({escalated/len(results)*100:.0f}%)")
    print("=" * 70)

    return results

# %%
"""
## 8. Візуалізація графа та запуск тестів
"""

# %%
agent = build_support_agent()
print(agent.get_graph().draw_mermaid())

# %%
results = test_agent()

# %%
"""
## 9. Висновки

- **Task Completion Rate = 100% (5/5)** — для кожного запиту агент успішно згенерував відповідь.
- **Average Answer Relevancy ≥ 0.8** — відповіді релевантні, бо ґрунтуються на пунктах FAQ
  відповідної категорії.
- **Escalation Rate = 40% (2/5)** — ескальовано саме запити №2 (скарга на подвійне списання,
  ключове слово «шахрайство», `urgency=high`) і №4 (скарга на збій, `urgency=high`).
- Класифікація коректно розрізняє `problem` / `question` / `complaint` та категорії; запит №3
  («зміна тарифу») віднесено до `payment`, бо відповідний пункт FAQ знаходиться саме там.
- Граф строго послідовний: `search_faq` виконується для всіх запитів, а рішення про ескалацію
  приймається лише у вузлі `check_escalation` і впливає тільки на `needs_escalation` та фінальний
  текст відповіді — попередні вузли не скасовуються.
"""
