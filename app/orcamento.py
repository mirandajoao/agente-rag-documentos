"""Limite diário de gasto da aplicação (FinOps).

Protege a conta da OpenAI quando o link é público: depois que o gasto do dia
passa do teto, novas perguntas, comparações, sugestões e uploads são recusados
até a virada do dia. O controle fica em memória (zera se o servidor reiniciar);
para um teto garantido, configure também o limite de gasto no painel da OpenAI.

Variáveis de ambiente:
    ORCAMENTO_DIARIO_USD   teto em US$ por dia (padrão 1.00; 0 desliga o limite)
    FUSO_ORCAMENTO         fuso que define a "virada do dia" (padrão America/Belem)
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from zoneinfo import ZoneInfo


class Orcamento:
    def __init__(self, limite_usd: float, fuso: str):
        self.limite = max(0.0, limite_usd)
        self.fuso = ZoneInfo(fuso)
        self._dia = self._hoje()
        self._gasto = 0.0
        self._lock = threading.Lock()

    def _hoje(self) -> str:
        return datetime.now(self.fuso).date().isoformat()

    def _virar_dia(self) -> None:
        hoje = self._hoje()
        if hoje != self._dia:
            self._dia, self._gasto = hoje, 0.0

    def excedido(self) -> bool:
        with self._lock:
            self._virar_dia()
            return bool(self.limite) and self._gasto >= self.limite

    def registrar(self, usd: float) -> None:
        with self._lock:
            self._virar_dia()
            self._gasto += max(0.0, float(usd or 0))

    def status(self) -> dict:
        with self._lock:
            self._virar_dia()
            return {"limite_usd": self.limite, "gasto_hoje_usd": round(self._gasto, 6), "dia": self._dia}


orcamento = Orcamento(float(os.getenv("ORCAMENTO_DIARIO_USD", "1.0")), os.getenv("FUSO_ORCAMENTO", "America/Belem"))

MENSAGEM_LIMITE = (
    "O limite diário de uso desta demonstração foi atingido. "
    "Tente novamente amanhã ou fale com os organizadores."
)
