"""Importar uma página da web (ou um PDF por link) como documento.

Segurança (SSRF): o servidor só baixa endereços públicos. Antes de cada
requisição (e de cada redirecionamento) o nome é resolvido e qualquer IP
privado, local, de loopback ou reservado é recusado; assim ninguém usa a
aplicação para chegar a serviços internos da rede do servidor.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

import httpx

MAX_BYTES = 8 * 1024 * 1024
MAX_REDIRECIONAMENTOS = 4
TIPOS_TEXTO = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown")


class LinkInvalido(ValueError):
    pass


def _checar_endereco(url: str) -> None:
    partes = urlparse(url)
    if partes.scheme not in {"http", "https"} or not partes.hostname:
        raise LinkInvalido("Use um link completo começando com http:// ou https://.")
    if partes.username or partes.password:
        raise LinkInvalido("Links com usuário e senha não são aceitos.")
    porta = partes.port or (443 if partes.scheme == "https" else 80)
    if porta not in {80, 443, 8080, 8443}:
        raise LinkInvalido("Só são aceitos links nas portas padrão da web.")
    try:
        enderecos = {info[4][0] for info in socket.getaddrinfo(partes.hostname, porta, proto=socket.IPPROTO_TCP)}
    except socket.gaierror as exc:
        raise LinkInvalido("Não encontrei esse endereço. Confira o link.") from exc
    for ip in enderecos:
        if not ipaddress.ip_address(ip.split("%")[0]).is_global:
            raise LinkInvalido("Esse endereço não é público e não pode ser acessado.")


def baixar(url: str) -> tuple[bytes, str, str]:
    """Baixa o link com segurança. Devolve (conteúdo, content-type, url final)."""
    url = url.strip()
    with httpx.Client(follow_redirects=False, timeout=15.0, headers={"User-Agent": "AgenteRAG-Oficina/1.0"}) as cliente:
        for _ in range(MAX_REDIRECIONAMENTOS + 1):
            _checar_endereco(url)
            with cliente.stream("GET", url) as resp:
                if resp.is_redirect:
                    destino = resp.headers.get("location")
                    if not destino:
                        raise LinkInvalido("O site redirecionou sem dizer para onde.")
                    url = urljoin(url, destino)
                    continue
                if resp.status_code >= 400:
                    raise LinkInvalido(f"O site respondeu com erro {resp.status_code}.")
                tipo = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if tipo != "application/pdf" and not tipo.startswith(TIPOS_TEXTO):
                    raise LinkInvalido(f"Esse link não é uma página de texto nem um PDF ({tipo or 'tipo desconhecido'}).")
                corpo = bytearray()
                for bloco in resp.iter_bytes():
                    corpo += bloco
                    if len(corpo) > MAX_BYTES:
                        raise LinkInvalido("A página passa de 8 MB.")
                return bytes(corpo), tipo, url
    raise LinkInvalido("Redirecionamentos demais.")


def html_para_texto(html: bytes, url: str) -> tuple[str, str]:
    """Extrai o texto principal de uma página. Devolve (título, texto em markdown simples)."""
    from bs4 import BeautifulSoup

    sopa = BeautifulSoup(html, "html.parser")
    for tag in sopa(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe"]):
        tag.decompose()
    titulo = (sopa.title.string or "").strip() if sopa.title and sopa.title.string else ""
    principal = sopa.find("article") or sopa.find("main") or sopa.body or sopa
    linhas: list[str] = []
    for el in principal.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "pre", "blockquote"]):
        texto = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not texto or (el.name in {"td", "th", "li"} and el.find(["p", "li"])):
            continue
        linhas.append(("# " if el.name in {"h1", "h2", "h3", "h4"} else "") + texto)
    texto = "\n\n".join(dict.fromkeys(linhas))
    if len(texto) < 200:  # páginas sem marcação semântica
        texto = re.sub(r"\n{3,}", "\n\n", principal.get_text("\n", strip=True))
    cabecalho = f"Fonte: {url}\n\n"
    return titulo or urlparse(url).hostname or "pagina", cabecalho + texto
