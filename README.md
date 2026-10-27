<p align="center">
  <img src="assets/banner.gif" alt="stochgame — stochastic games & game theory for AI agents" width="100%">
</p>

<p align="center">
  <a href="https://github.com/vantacorehq/stochgame/actions/workflows/ci.yml"><img src="https://github.com/vantacorehq/stochgame/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/deps-numpy%20only-brightgreen" alt="numpy only">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT"></a>
</p>

<p align="center">
  <b>Русский</b> · <a href="README.en.md">English</a>
</p>

# stochgame

Движок стохастических игр и симуляция теории игр для автономных ИИ-агентов.
**Один файл, одна зависимость — `numpy`.**

<img src="assets/divider.gif" width="100%" alt="">

## Быстрый старт

```bash
pip install numpy
git clone https://github.com/vantacorehq/stochgame.git
cd stochgame
python stochgame.py selftest   # 8 проверок
python stochgame.py demo       # экскурсия по всему (~20 с)
```

Или просто скопируйте `stochgame.py` в свой проект — больше ничего не нужно.

## Что внутри

| Блок             | Что есть                                                                                                           |
|------------------|--------------------------------------------------------------------------------------------------------------------|
| Решатели         | точный LP для антагонистических игр (симплекс), все равновесия Нэша биматричной игры (перебор носителей), `nash_gap` |
| `StochasticGame` | N игроков, S состояний, `P[s,a0..aN,s']`, `R[s,a0..aN,i]`, терминальные состояния                                  |
| Планирование     | итерация Шепли (`shapley`) для 2-игроковых zero-sum игр                                                            |
| Агенты           | `RandomAgent`, `Always`, `QLearner`, `MinimaxQ`, `RegretMatcher`, `TitForTat`, `GrimTrigger`, `Pavlov`             |
| Симуляция        | `play`, `tournament` (round-robin), `replicator` (RK4)                                                             |
| Игры             | дилемма заключённого, охота на оленя, chicken, камень-ножницы-бумага, орёл-решка, повторяющиеся игры с памятью, `commons` (трагедия общин), `random_game` |

## Примеры

### Два Q-learner'а в «трагедии общин»

```python
import stochgame as sg

g = sg.commons()                                   # стохастическая игра с общим ресурсом
agents = [sg.QLearner(), sg.QLearner()]
res = sg.play(g, agents, episodes=300, horizon=50)
print(res.returns[-30:].mean(0), res.freq[-30:, :, 0].mean())   # доход, доля «бережного» действия
```

### Турнир в повторяющейся дилемме заключённого

```python
names, M = sg.tournament(
    sg.prisoners_dilemma(),
    {"TFT": sg.TitForTat, "ALLD": lambda: sg.Always(1), "Pavlov": sg.Pavlov},
)
print(sg.format_matrix(names, M))
```

### Точное решение zero-sum стохастической игры

```python
V, policies = sg.shapley(sg.random_game(4, 3, 3, seed=7))
```

### Равновесия Нэша биматричной игры

```python
for x, y in sg.nash_equilibria(sg.CHICKEN, sg.CHICKEN.T):
    print(x.round(2), y.round(2))
```

### Свой агент

Наследуйте `Agent`, реализуйте `act(s)` и (по желанию) `observe(s, acts, r, s2, done)`:

```python
class Sulky(sg.Agent):
    """Сотрудничает, пока оппонент не предал два раза подряд."""
    name = "Sulky"

    def reset(self): self.bad = 0
    def new_episode(self): self.bad = 0
    def act(self, s): return 1 if self.bad >= 2 else 0
    def observe(self, s, acts, r, s2, done):
        self.bad = self.bad + 1 if acts[1 - self.me] == 1 else 0

names, M = sg.tournament(sg.prisoners_dilemma(), {"TFT": sg.TitForTat, "Sulky": Sulky})
```

Свою игру — передайте тензоры `P` и `R` в `StochasticGame`.

## Структура репозитория

```
stochgame.py          # весь движок (один файл)
tests/                # pytest
tools/make_banner.py  # генератор баннеров (numpy + stdlib)
assets/               # banner.gif, banner.png, divider.gif
.github/workflows/    # CI
```

## Баннеры

`python tools/make_banner.py` пересоздаёт `assets/banner.gif`, `banner.png`, `divider.gif`.
Тоже только numpy + stdlib: свой растровый шрифт, свои GIF (LZW) и PNG энкодеры.
Кометы на баннере — реальные орбиты replicator dynamics для игры «камень-ножницы-бумага»,
посчитанные самим движком.

## Разработка

```bash
pip install numpy pytest
pytest -q
```

## Ссылки

- Shapley (1953), *Stochastic games*
- Littman (1994), *Markov games as a framework for multi-agent reinforcement learning*
- Axelrod (1984), *The Evolution of Cooperation*

## Лицензия

[MIT](LICENSE)
