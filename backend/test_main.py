from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Garante que a API está online e respondendo."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_empty_question():
    """Garante que o Pydantic bloqueie perguntas vazias antes de bater na IA (Erro 422)."""
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422

def test_chat_valid_question():
    """Garante que o chat responde corretamente a uma pergunta válida (Status 200)."""
    response = client.post("/chat", json={"question": "Quantos acidentes ocorreram na BR-101?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert type(data["sources"]) == list