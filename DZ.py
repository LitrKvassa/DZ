from copy import deepcopy
from itertools import product
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import click
from noise import pnoise2

data = {
    # База
    'N': 100, 'M': 100, 'p_fire': .01, 'p_grow': .01, 
    'trees_to_ign': 1, 't': 109, 'inter': 100, 'areatype': 1,
    
    # Параметры влажности
    'H_diff_por': 0.05,    # Минимальный порог разницы влажности для перетекания
    'p_rain': .5,         # Вероятность дождя
    'p_rain_check': 10,        # Раз во сколько шагов может пойти дождь
    'p_grow_H_w': 0.01,       # влияние влажности на рост дерева
    'p_tree_H_gen': 0.05, # Вероятность выделения влаги деревом
    'tree_H_gen': 0.1, # количество выделяемой деревом влаги
    # значение выд. влаги равно wind_H_transfer
    # Параметры ветра
    'wind_v': 1.0,        # Сила ветра (0 - выключен)
    'wind_dir': 90.0,         # Направление ветра в градусах (0 - вверх, 90 - вправо...)
    'wind_w': 1.,            # Весовой коэффициент влияния ветра
    'wind_H_transfer': 0.1     # перенос влажности ветром
}

def show_menu():
    click.clear()
    click.secho("=== КА лесного пожара ===", bold=True)
    click.echo("1. Текущие параметры")
    click.echo("2. Ввод параметров")
    click.echo("3. Расчёт КА")
    click.echo("4. Выход")
    click.echo("=======================")

def show_data():
    click.clear()
    click.secho("--- ТЕКУЩИЕ ПАРАМЕТРЫ ---", bold=True)
    for key, value in data.items():
        click.echo(f" {key} = {value}")
    click.echo("-------------------------")
    input("<ENTER для возврата в меню>")

def input_data():
    click.clear()
    click.secho("--- ВВОД ДАННЫХ ---", bold=True)
    click.echo("\nENTER без ввода значения чтобы оставить текущее значение>\n")

    # Базовые параметры
    data['N'] = click.prompt("N (число строк, int)", default=data['N'], type=click.IntRange(min=2))
    data['M'] = click.prompt("M (число столбцов, int)", default=data['M'], type=click.IntRange(min=2))
    data['t'] = click.prompt("t (число итераций, int)", default=data['t'], type=click.IntRange(min=1))
    data['inter'] = click.prompt("inter (задержка анимации, мс, int)", default=data['inter'], type=click.IntRange(min=1))
    data['trees_to_ign'] = click.prompt("trees_to_ign (число горящих деревьев для переноса огня, int)", default=data['trees_to_ign'], type=click.IntRange(min=1))
    data['areatype'] = click.prompt("areatype (окрестность: 0 - Неймана, 1 - Мура, int)", default=data['areatype'], type=click.IntRange(min=0, max=1))
    data['p_fire'] = click.prompt("p_fire (вероятность случайного возгорания, float)", default=data['p_fire'], type=click.FloatRange(min=0.0, max=1.0))
    data['p_grow'] = click.prompt("p_grow (вероятность роста дерева, float)", default=data['p_grow'], type=click.FloatRange(min=0.0, max=1.0))
    click.secho("\n")
    # Параметры влажности
    data['H_diff_por'] = click.prompt("H_diff_por (мин. разница вл. воздуха клеток для переноса, float)", default=data['H_diff_por'], type=click.FloatRange(min=0.0, max=1.0))
    data['p_grow_H_w'] = click.prompt("p_grow_H_w (влияние влажности на рост дерева, float)", default=data['p_grow_H_w'], type=click.FloatRange(min=0.0, max=1.0))
    data['p_tree_H_gen'] = click.prompt("p_tree_H_gen (вероятность повышения уровня влажности деревом, float)", default=data['p_tree_H_gen'], type=click.FloatRange(min=0.0, max=1.0))
    data['tree_H_gen'] = click.prompt("tree_H_gen (количество выделяемой деревом влаги, float)", default=data['p_tree_H_gen'], type=click.FloatRange(min=0.0, max=1.0))
    data['p_rain'] = click.prompt("p_rain (вероятность выпадения дождя, float)", default=data['p_rain'], type=click.FloatRange(min=0.0, max=1.0))
    data['p_rain_check'] = click.prompt("p_rain_check (периодичность выпадения дождя, шагов, int)", default=data['p_rain_check'], type=click.IntRange(min=1))
    click.secho("\n")
    # Параметры ветра
    data['wind_v'] = click.prompt("wind_v (сила ветра (0 - выкл.), float)", default=data['wind_v'], type=click.FloatRange(min=0.0, max=1.))
    data['wind_dir'] = click.prompt("wind_dir (направление ветра, градусов, float)", default=data['wind_dir'], type=click.FloatRange(min=0.0, max=360.0))
    data['wind_w'] = click.prompt("wind_w (коэф. влияния ветра на распространение огня и перенос влаги, float)", default=data['wind_w'], type=click.FloatRange(min=0.0, max = 1.))
    data['wind_H_transfer'] = click.prompt("wind_H_transfer (количество переносимой ветром влаги, float)", default=data['wind_H_transfer'], type=click.FloatRange(min=0.0, max=1.0))
    
    click.secho("\n<Данные обновлены>")
    input("<ENTER для возврата в меню>")

def area_calc(i, j, rows, cols, areatype):
    area = []
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if areatype == 1:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for di, dj in offsets:
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            area.append((ni, nj, di, dj))
    return area

def updFH(ca, H, step_num):
    rows, cols = len(ca), len(ca[0])
    new_F = deepcopy(ca)
    new_H = deepcopy(H)
    areatype = data['areatype']
    H_diff = data['H_diff_por']
    p_grow_H_w = data['p_grow_H_w']
    wind_v = data['wind_v']
    wind_dir_rad = np.radians(data['wind_dir'])
    wind_w = data['wind_w']
    wind_H_transfer = data['wind_H_transfer']
    p_tree_H_gen = data['p_tree_H_gen']
    tree_H_gen = data['tree_H_gen']
    
    wind_i = -np.cos(wind_dir_rad) if wind_v > 0 else 0
    wind_j = np.sin(wind_dir_rad) if wind_v > 0 else 0

    # Диффузия влаги
    for i, j in product(range(rows), range(cols)):
        area = area_calc(i, j, rows, cols, areatype)
        if not area:
            continue

        receivers = []
        for ni, nj, _, _ in area:
            if H[i][j] - H[ni][nj] > H_diff:
                receivers.append((ni, nj))
        
        if receivers:
            total_diff = sum(H[i][j] - H[ni][nj] for ni, nj in receivers)
            share = total_diff / len(receivers) * 0.1
            for ni, nj in receivers:
                new_H[ni][nj] += share / len(receivers)
                new_H[i][j] -= share / len(receivers)

        # Направленный перенос влаги
        if wind_v > 0 and H[i][j] > 0.01:
            total_wind_weight = 0
            valid_receivers = []

            for ni, nj, di, dj in area:
                mag = np.hypot(di, dj)
                dot = (di/mag * wind_i) + (dj/mag * wind_j)
                
                if dot > 0 and (H[i][j] - H[ni][nj] > H_diff):
                    valid_receivers.append((ni, nj, dot))
                    total_wind_weight += dot
            
            if valid_receivers:
                total_amount = H[i][j] * wind_H_transfer * wind_w

                max_safe_transfer = H[i][j] - min(H[ni][nj] for ni, nj, _ in valid_receivers) - H_diff
                total_amount = min(total_amount, max(0.0, max_safe_transfer))
                
                if total_amount > 0:
                    new_H[i][j] -= total_amount
                    for ni, nj, dot in valid_receivers:
                        share = total_amount * (dot / total_wind_weight)
                        new_H[ni][nj] += share

        if ca[i][j] == 1 and rnd.random() < p_tree_H_gen:
            if new_H[i][j] < 0.75:
                new_H[i][j] = min(new_H[i][j] + tree_H_gen, 0.75)

    new_H = np.clip(new_H, 0.0, 1.0)
    H = deepcopy(new_H)

    # Дождь 
    if step_num > 0 and step_num % data['p_rain_check'] == 0 and rnd.random() < data['p_rain']:
        num_rain_cells = int(rows * cols * 0.1)
        for _ in range(num_rain_cells):
            ri = rnd.randint(0, rows - 1)
            rj = rnd.randint(0, cols - 1)
            new_H[ri][rj] = rnd.uniform(0.9, 0.99)

    # Обновление состояний автомата
    for i, j in product(range(rows), range(cols)):
        area = area_calc(i, j, rows, cols, areatype)
        burn_count = 0
        fire_ps = []
        
        for ni, nj, di, dj in area:
            if ca[ni][nj] == 2:
                burn_count += 1
                if wind_v > 0:
                    mag = np.hypot(-di, -dj)
                    dot = ((-di/mag) * wind_j) + ((-dj/mag) * wind_i)
                    wind_mod = 1.0 + wind_w * wind_v * dot
                else:
                    wind_mod = 1.0
                
                H_mod = 1.0 - H[i][j]
                fire_ps.append(data['p_fire'] * wind_mod * H_mod)
        # Горение
        if ca[i][j] == 2:
            new_F[i][j] = 0  
        elif ca[i][j] == 1:
            ign = False
            #1
            if burn_count >= data['trees_to_ign']:
                avg_wind_mod = 1.0
                if wind_v > 0:
                    wind_mods = []
                    for ni, nj, di, dj in area:
                        if ca[ni][nj] == 2:
                            mag = np.hypot(-di, -dj)
                            dot = ((-di/mag) * wind_j) + ((-dj/mag) * wind_i)
                            wind_mods.append(1.0 + wind_w * wind_v * dot)
                    if wind_mods:
                        avg_wind_mod = np.mean(wind_mods)

                H_mod = 1.0 - H[i][j]
                if areatype == 0:
                    excess_factor = 1.0 + (burn_count - data['trees_to_ign'])/(4-burn_count)
                else:
                    excess_factor = 1.0 + (burn_count - data['trees_to_ign'])/(8-burn_count)
                p_critical_fire = min(1.0, excess_factor * avg_wind_mod * H_mod)
                
                if rnd.random() < p_critical_fire:
                    ign = True
            #2
            else:
                for p_fire_mod in fire_ps:
                    if rnd.random() < p_fire_mod:
                        ign = True
                        break
            #rnd 3
                if not ign and rnd.random() < (data['p_fire'] * (1.0 - H[i][j])):
                    ign = True

            if ign:
                new_F[i][j] = 2
                new_H[i][j] = 0.0
        # Рост
        elif ca[i][j] == 0 and burn_count == 0:
            p_grow = data['p_grow'] + p_grow_H_w * H[i][j]
            if rnd.random() < min(p_grow, 1.0):
                new_F[i][j] = 1

    return new_F, new_H

def upd0(ca, rng):
    rows, columns = range(len(ca)), range(len(ca[0]))
    for i, j in product(rows, columns):
        ca[i][j] = rng.integers(0, 2)
    return ca

def upd0H(rows, cols, scale=20.0):
    H = np.zeros((rows, cols))
    base_x = rnd.uniform(0, 1000)
    base_y = rnd.uniform(0, 1000)
    for i in range(rows):
        for j in range(cols):
            noise_val = pnoise2(base_x + i / scale, base_y + j / scale, octaves=2, persistence=0.5)
            H[i][j] = (noise_val + 1.0) / 2.0
    return H

def run_calc():
    click.clear()
    click.secho(">")
    
    N, M = data['N'], data['M']
    t = data['t']
    inter = data['inter']
    
    F = np.zeros((N, M))
    rng = np.random.default_rng()
    
    plt.close('all')
    F = upd0(F, rng)
    H = upd0H(N, M)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

    colors_list = ['tan', 'green', 'orange']
    bounds = [-0.5, 0.5, 1.5, 2.5]
    cmap_fire = mcolors.ListedColormap(colors_list)
    norm_fire = mcolors.BoundaryNorm(bounds, cmap_fire.N)
    
    im_fire = ax1.imshow(F, cmap=cmap_fire, norm=norm_fire, aspect='auto', origin='upper')
    ax1.set_title("Модель пожара (Шаг: 0)")
    ax1.grid(visible=True, which='major', color='black', linestyle='-', linewidth=.25)
    cbar1 = plt.colorbar(im_fire, ax=ax1, ticks=[0, 1, 2])
    cbar1.ax.set_yticklabels(['Пустая клетка', 'Дерево', 'Огонь'])

    im_hum = ax2.imshow(H, cmap='Blues', aspect='auto', origin='upper', vmin=0, vmax=1)
    ax2.set_title("Карта влажности воздуха")
    ax2.grid(False)
    cbar2 = plt.colorbar(im_hum, ax=ax2)
    cbar2.set_label('Уровень влаги')
    
    for ax in [ax1, ax2]:
        ax.set_xlabel("Столбцы")
        ax.set_ylabel("Строки")
        
    step = -1
    
    def update_frame(frame):
        nonlocal F, H, step
        step += 1
        F, H = updFH(F, H, step)
        im_fire.set_data(F)
        im_hum.set_data(H)
        ax1.set_title(f"Модель пожара (Шаг: {step})")
        return [im_fire, im_hum]

    ani = FuncAnimation(fig, update_frame, frames=t, interval=inter, blit=False)
    ani.save('animation.gif', writer='pillow')
    
    flat_F = F.flatten()
    zeros = np.sum(flat_F == 0)
    ones = np.sum(flat_F == 1)
    fires = np.sum(flat_F == 2)
    
    print(f'\nПустые клетки — {zeros}, Деревья — {ones}, Горящие клетки — {fires}')
    print(f'Средний уровень влажности воздуха по карте: {np.mean(H):.2f}\n')
    click.secho("Сохранено как 'animation.gif'\n")
    plt.show(fig)
    plt.close(fig)
    input("<ENTER для возврата в меню>")

@click.command()
def main():
    while True:
        show_menu()
        choice = click.prompt("Ввод", type=click.Choice(['1', '2', '3', '4']), show_choices=False)
        if choice == '1':
            show_data()
        elif choice == '2':
            input_data()
        elif choice == '3':
            run_calc()
        elif choice == '4':
            click.secho("\nРабота программы завершена")
            break

if __name__ == '__main__':
    main()