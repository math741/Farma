# Instruções para o Claude

- **Responda sempre em português do Brasil** (pedido permanente do usuário).

## Projeto

GENESIS Ω: pacote Python `genesis_omega` (simulação em PyTorch, servidor aiohttp, observatório
three.js em `genesis_omega/viewer/static`).

- Rodar localmente: `python -m genesis_omega run --profile cpu --no-tunnel`
- Testes: `python -m pytest` (perfil `cpu-test`; o teste do navegador precisa do Playwright)
- Regra da casa: qualquer mudança nas leis do universo (física, química, corpo) exige subir
  `PHYSICS_VERSION` em `genesis_omega/version.py`; o registro honesto de funcionalidades fica em
  `genesis_omega/core/status.py` e nunca deve afirmar mais do que existe.
