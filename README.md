# GENESIS Ω — O Cosmos Vivo

Um universo artificial **persistente**: galáxia procedural, planetas com clima, organismos que
crescem a partir de um genoma de desenvolvimento e controlam o corpo com um substrato de sinais
plástico, química de materiais aberta (escondida), física de construção, um **Cientista**
observador que só aceita resultados que sobrevivem a controles, replicação e ablação — e um
**observatório 3D** no navegador (WebGL2 / WebXR).

Nada neste pacote declara vida, consciência, sentimentos ou civilização. Ele **simula, mede e
rotula** — e diz o que é o quê: cada afirmação carrega uma etiqueta epistêmica (`SIMULATED`,
`PROGRAMMED`, `EMERGENT`, `MEASURED`, `INFERRED`, `HYPOTHESIS`, `EXTERNAL`) e, quando é ciência,
um status de validação (`CANDIDATE` → `TESTED` → `REPLICATED` → `CAUSALLY SUPPORTED`, ou `REJECTED`).

## Começo rápido

Requisitos: Python ≥ 3.10, PyTorch ≥ 2.1, NumPy, SciPy, aiohttp.

```bash
pip install -e .
```

```bash
python -m genesis_omega run --profile cpu --no-tunnel
```

O terminal imprime dois links:

- `local_viewer`: somente leitura, pode ser compartilhado;
- `local_admin`: permite forks, intervenções e checkpoints.

Abra um deles no navegador e clique em **ENTER COSMOS**. `Ctrl+C` grava um checkpoint e para.

Para ver o hardware detectado e o registro honesto de funcionalidades:

```bash
python -m genesis_omega info
```

### Google Colab

Abra `GenesisOmega.ipynb` no Colab e use **Executar tudo**. O notebook monta o Google Drive, onde o
cosmos persiste entre sessões, inicia a simulação e o servidor, e imprime links públicos por um
túnel Cloudflare. Na próxima sessão, o mesmo cosmos é **retomado**, nunca recriado.

## Perfis de hardware

O perfil é escolhido pelo hardware detectado; nada assume uma A100. Num cosmos novo, o
autoajuste mede o desempenho e ajusta só a capacidade (número de organismos), nunca as leis.

| perfil     | dispositivo | estrelas | mundo F4  | organismos | uso                    |
|------------|-------------|----------|-----------|------------|------------------------|
| `cpu-test` | CPU         | 400      | 64×32     | 128        | testes                 |
| `cpu`      | CPU         | 3 000    | 160×80    | 512        | notebook / laptop      |
| `gpu`      | CUDA        | 12 000   | 384×192   | 4 096 (+2º mundo) | GPUs comuns     |
| `a100`     | CUDA        | 20 000   | 512×256   | 12 288 (+2º mundo) | ≥ 70 GB de VRAM |

## O observatório

- **Galáxia**: cada ponto é uma estrela simulada. O halo verde marca biosferas agregadas (F0) e o
  anel ciano marca mundos com indivíduos (F4). No centro fica o buraco negro supermassivo.
- **Sistema**: órbitas keplerianas com os elementos reais e a zona habitável. O céu ao fundo são as
  outras estrelas da galáxia vistas daquele sistema.
- **Planeta**: tudo vem do estado da simulação:
  - relevo e oceanos reais;
  - cor da superfície a partir da temperatura, produção primária, umidade e trilhas;
  - gelo marinho abaixo de 0 °C;
  - nuvens a partir do campo de umidade;
  - dia e noite pelo ponto subsolar;
  - brilho noturno que mostra a ocupação medida de organismos;
  - estruturas construídas pelos próprios organismos.

  Camadas alternativas: temperatura, produção, umidade, ocupação, estruturas, trilhas, abrigo e
  marcas.
- **Organismo**: clique num ponto do planeta para abrir o inspetor, que mostra:
  - energia, integridade e temperatura corporal;
  - o corpo crescido a partir do genoma (voxels em 3D);
  - o estado do substrato de sinais, os sinais de entrada e os efetores;
  - ancestralidade e laços sociais.

  O botão "seguir" acompanha o organismo com a câmera.
- **Painéis**:
  - *História*: eventos, panteão das "primeiras vezes" e o que aconteceu desde a sua última visita;
  - *Ciência*: controles, descobertas e níveis avaliados por evidência;
  - *Vida*: filogenia, nichos MAP-Elites e grafo social;
  - *Tecnologia*: grafo de materiais, linhagem cultural e detectores de assentamento;
  - *Status*: sala de máquinas e registro de funcionalidades;
  - *Admin*: forks, intervenções, trava da linha principal, checkpoint, verificação de replay e
    ponte de realidade.
- **Linha do tempo**: todos os eventos no eixo de ciclos. Clique num evento para inspecioná-lo;
  eventos históricos guardam uma janela gravada que pode ser **reproduzida**.
- **WebXR**: botões VR e PORTAL (AR) aparecem quando o dispositivo suporta. A sonificação (♪) não
  é fala: mapeia temperatura, população e emissões em som.

## Experimentos: a DSL do Cientista

A linha do tempo principal (PRIME) é protegida: experimentos rodam em **forks**, e cada ação é um
evento `EXTERNAL` naquela linha do tempo. Os experimentos são documentos JSON de vocabulário
fechado: nada é avaliado, executado ou importado.

```json
{"world": 0, "steps": [
  {"op": "ablate", "mechanism": "vibration"},
  {"op": "perturb", "param": "solar_flux", "value": 1.05},
  {"op": "inject", "field": "light", "lat": 10, "lon": 20, "gain": 2},
  {"op": "cull", "fraction": 0.3},
  {"op": "observe", "metrics": ["population", "mean_energy", "births"]}
]}
```

| parte       | valores aceitos |
|-------------|-----------------|
| operações   | `fork`, `ablate`, `perturb`, `observe`, `inject`, `cull`, `expedition` |
| mecanismos  | `vibration`, `plasticity`, `chemistry`, `marks`, `scent`, `material` |
| parâmetros  | `solar_flux`, `growth_rate`, `move_scale`, `sound_decay` (com limites numéricos) |

## Persistência

```
<raiz>/<rótulo-do-cosmos>/
    cosmos.json              identidade, semente, versões (nunca reescrita em silêncio)
    auth.json                tokens de leitura / admin (privado)
    checkpoints/             current.pt, previous.pt, history/  (+ .json com sha256)
    events/events.jsonl      história, só acrescenta
    discoveries/             pacotes de descobertas + janelas congeladas (.npz)
    forks/  reports/  failures/  telemetry/
```

As escritas são atômicas e verificadas por SHA-256, com recuo automático:
current → previous → histórico. Um cosmos sem checkpoint válido **nunca** é substituído em
silêncio: o reinício exige `--reset`, e mesmo assim o cosmos antigo é arquivado, nunca apagado.
Uma mudança nas leis (`PHYSICS_VERSION`) não continua a linha do tempo principal sem uma migração
explícita.

## Estrutura do código

```
genesis_omega/
  core/      configuração, hardware, RNG determinístico, ids, logs, registro de funcionalidades
  cosmos/    galáxia, estrelas, sistemas planetários, biosferas F0
  world/     mundo F4 (o tick), geometria esférica, terreno, clima, física, química de materiais
  life/      genoma, desenvolvimento → corpo, substrato de sinais, agentes, evolução
  society/   grafo social, comunicação, construção, tecnologia
  science/   Cientista, sondas, controles, estatística, DSL de experimentos, níveis
  history/   eventos, crônica, panteão
  storage/   checkpoints atômicos e migrações
  runtime/   universo, laço de simulação, autoajuste, lançador
  server/    HTTP + WebSocket (aiohttp), protocolo binário GOB1, túnel
  viewer/    observatório (three.js r170 vendorizado, módulos ES sem build)
tests/       suíte pytest (incluindo o visualizador num Chromium headless)
```

## Testes

```bash
pip install -e ".[test]"
```

```bash
python -m pytest
```

O teste do visualizador (`tests/test_viewer_headless.py`) usa Playwright com Chromium. Se ele não
estiver instalado, o teste é pulado. Para instalá-lo:

```bash
playwright install chromium
```

## O que existe (e o que não existe)

O registro em `genesis_omega/core/status.py` é a fonte única da verdade e aparece no painel
*Status*. Cada funcionalidade é `TESTED`, `UNVALIDATED` ou `NOT IMPLEMENTED`. Não existem, por
exemplo:

- motor de voo espacial;
- primeiro contato;
- níveis de afeto A5–A9;
- testes de composicionalidade e de aprendizagem social;
- renderizador WebGPU.

Nenhum nível mede consciência ou sentimento.
