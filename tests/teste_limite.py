import os, sys, tempfile
from pathlib import Path
REPO = Path(sys.argv[1]); tmp = Path(tempfile.mkdtemp())
os.environ.update(OPENAI_API_KEY="sk-teste", CHROMA_DIR=str(tmp/"c"), UPLOAD_DIR=str(tmp/"u"))
sys.path.insert(0, str(REPO))
from fastapi import HTTPException
from starlette.requests import Request
import app.api as api
req = Request({"type": "http", "headers": [(b"x-forwarded-for", b"200.1.1.1")], "client": ("200.1.1.1", 1)})
def tenta(chave, upload=False):
    try: api._limitar(req, chave, upload); return True
    except HTTPException: return False
# 30 alunos no mesmo IP, 10 perguntas cada em 1 minuto
ok = sum(tenta(f"aluno{i:02d}xyz") for i in range(30) for _ in range(10))
print("30 alunos × 10 perguntas, mesmo IP:", ok, "de 300 permitidas")
print("16ª pergunta do mesmo aluno:", [tenta("alunoUNICO") for _ in range(16)][-1])
up = sum(tenta(f"aluno{i:02d}xyz", True) for i in range(30) for _ in range(3))
print("30 alunos × 3 uploads:", up, "de 90 permitidos")
assert ok == 300 and up == 90
print("OK")
