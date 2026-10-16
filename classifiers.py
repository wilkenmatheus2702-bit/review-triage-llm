"""Classifiers: an LLM (Claude) and a rule-based baseline, behind one interface.

Every classifier returns a :class:`Prediction`. Invalid LLM output never
crashes the run: it is retried once with the validation error, and if it is
still invalid the review is routed to a human.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from .prompt import system_prompt, user_message
from .schema import ReviewLabel, tool_schema

# USD per million tokens (input, output), standard tier.
# Source: platform.claude.com/docs/en/about-claude/pricing (checked Oct 2026).
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "rules": (0.0, 0.0),
}
DEFAULT_MODEL = "claude-haiku-5-5"
TOOL_NAME = "record_triage"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


@dataclass
class Prediction:
    label: ReviewLabel | None
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.input_tokens, self.output_tokens)


class Classifier(Protocol):
    name: str
    uses_cache: bool

    def classify(self, text: str) -> Prediction: ...


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------
class LLMClassifier:
    """Claude with a forced tool call, so the answer always comes back as JSON
    in the shape of :func:`tool_schema`, then validated by pydantic."""

    uses_cache = True

    def __init__(self, model: str = DEFAULT_MODEL, client=None, validation_retries: int = 1):
        if client is None:
            import anthropic  # imported lazily: the baseline works without the SDK or a key

            client = anthropic.Anthropic(max_retries=4)  # SDK retries 429/5xx with backoff
        self.client = client
        self.model = model
        self.name = model
        self.validation_retries = validation_retries
        self._system = system_prompt()
        self._tool = {
            "name": TOOL_NAME,
            "description": "Record the structured triage of one customer review.",
            "input_schema": tool_schema(),
        }

    def _call(self, messages: list[dict]):
        return self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=self._system,
            tools=[self._tool],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=messages,
        )

    def classify(self, text: str) -> Prediction:
        messages: list[dict] = [{"role": "user", "content": user_message(text)}]
        tokens_in = tokens_out = 0
        error = "no answer"
        for _attempt in range(self.validation_retries + 1):
            try:
                response = self._call(messages)
            except Exception as exc:  # network/auth errors after SDK retries
                return Prediction(None, self.model, tokens_in, tokens_out, f"api_error: {exc}")
            tokens_in += response.usage.input_tokens
            tokens_out += response.usage.output_tokens
            block = next((b for b in response.content if b.type == "tool_use"), None)
            if block is None:
                error = "model did not call the tool"
                break
            try:
                label = ReviewLabel.model_validate(block.input)
                return Prediction(label, self.model, tokens_in, tokens_out)
            except ValidationError as exc:
                error = f"invalid_output: {exc.errors()[0]['msg']}"
                # Show the model its mistake and ask again.
                messages += [
                    {
                        "role": "assistant",
                        "content": [block.model_dump() if hasattr(block, "model_dump") else dict(block)],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "is_error": True,
                                "content": f"Invalid input: {exc}. Call {TOOL_NAME} again with valid values.",
                            }
                        ],
                    },
                ]
        return Prediction(None, self.model, tokens_in, tokens_out, error)


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------
def normalise(text: str) -> str:
    """Lowercase and strip accents, so 'não', 'nao' and 'NÃO' all match."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


RULES: dict[str, re.Pattern] = {
    "wrong_or_missing_item": re.compile(
        r"\b(so|somente|apenas) (veio|recebi|chegou|entregaram|foi entregue|foram entregues)"
        r"|falt(ou|ando|a) |incomplet|nao veio (o|a|as|os) |veio errad|produto errad|errad[oa]\b"
        r"|diferente do que (comprei|pedi)|trocaram|outra cor|cor errada|nao (e|era) o que (comprei|pedi)"
    ),
    "not_received": re.compile(
        r"nao (recebi|recebemos|chegou|foi entregue|entregue|foi entregu)|ainda nao (recebi|chegou)"
        r"|aguardando|aguardo|nao recebi|cade (o|meu)|ate (agora|hoje|o momento) (nada|nao)"
    ),
    "late_delivery": re.compile(r"atras|demor|fora do prazo|passou do prazo|nao (foi )?cumprid"),
    "damaged_or_defective": re.compile(
        r"defeit|quebrad|amassad|riscad|sujo|suja\b|nao funciona|avaria|estourad|danificad|violad|manchad"
    ),
    "quality_or_description": re.compile(
        r"qualidade (ruim|inferior|pessima|baixa|nao)|diferente da (foto|imagem)|inferior|pequen"
        r"|propaganda enganosa|nao (e|era) (o que|como)|nao corresponde|esperava mais|nao serv"
    ),
    "seller_service": re.compile(
        r"(sem|nenhum|nao (tive|obtive|houve|tenho)) (retorno|resposta|informac|explicac|satisfac)"
        r"|nao respond|ninguem|atendimento (pessimo|ruim)|pessimo atendimento|nao consigo contato"
    ),
    "return_or_refund": re.compile(
        r"devol|reembols|estorn|cancel|dinheiro de volta|\btroca|ressarc|restitui"
    ),
}
PRAISE = re.compile(
    r"otim|excelente|recomendo|adorei|gostei|perfeit|maravilh|parabens|satisfeit|muito bom|muito boa"
    r"|lindo|linda|rapid|antes do prazo|no prazo|dentro do prazo|confiav|tudo certo|tudo ok|amei|bom\b|boa\b"
)
NEGATED_PRAISE = re.compile(r"nao (recomendo|gostei|e bom|e boa|adorei|e otim|foi rapid)")
ACTION_TOPICS = {
    "not_received",
    "wrong_or_missing_item",
    "damaged_or_defective",
    "return_or_refund",
    "seller_service",
}


class RuleClassifier:
    """Keyword baseline. Free and instant; the bar the LLM has to beat."""

    name = "rules"
    uses_cache = False

    def classify(self, text: str) -> Prediction:
        t = normalise(text)
        topics = {name for name, pattern in RULES.items() if pattern.search(t)}
        if "not_received" in topics and "wrong_or_missing_item" in topics:
            topics.discard("not_received")  # "nao veio o espeto" is a missing item
        if "not_received" in topics:
            topics.discard("late_delivery")
        praise = bool(PRAISE.search(NEGATED_PRAISE.sub(" ", t)))
        problems = set(topics)
        if praise:
            topics.add("praise")
        if not topics:
            topics.add("other")
        if problems and not praise:
            sentiment = "negative"
        elif praise and not problems:
            sentiment = "positive"
        elif NEGATED_PRAISE.search(t) or re.search(r"pessim|horrivel|lament|nunca mais|roub", t):
            sentiment = "negative"
        else:
            sentiment = "neutral"
        label = ReviewLabel(
            sentiment=sentiment,
            topics=sorted(topics)[:4],
            needs_action=bool(problems & ACTION_TOPICS),
            summary_en="(rules baseline: no summary)",
            confidence=0.9 if len(topics) == 1 and "other" not in topics else 0.6,
        )
        return Prediction(label, self.name)


# ---------------------------------------------------------------------------
# Replay classifier (tests and demos without an API key)
# ---------------------------------------------------------------------------
class ReplayClassifier:
    """Returns pre-recorded answers keyed by review text."""

    uses_cache = True

    def __init__(self, answers: dict[str, ReviewLabel | Exception], name: str = "replay"):
        self.answers = answers
        self.name = name
        self.calls = 0

    def classify(self, text: str) -> Prediction:
        self.calls += 1
        answer = self.answers.get(text)
        if isinstance(answer, ReviewLabel):
            return Prediction(answer, self.name, 500, 60)
        return Prediction(None, self.name, 500, 60, str(answer or "no recorded answer"))
