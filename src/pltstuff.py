import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

results_dir = 'data/testresults'
csv_files = sorted(glob.glob(os.path.join(results_dir, '*.csv')), key=lambda x: int(os.path.basename(x).replace('.csv', '')))

colors = ['#534AB7', '#185FA5', '#0F6E56', '#639922', '#BA7517', '#D85A30', '#A32D2D', '#D4537E', '#1D9E75', '#888780']

test_labels = {
    1: '3 Adversaries, including two designed to trick the Agent',
    2: '4 Adversaries, including itself and random',
    3: '4 Adversaries, including itself and two purpose-designed Agents',
    4: '2 Adversaries, including random',
    5: '5 Adversaries, including itself and two purpose-designed Agents',
    6: '3 Adversaries, including random',
    7: '7 Adversaries, including itself, random, and two purpose-designed Agents'
}

final_rates = []
test_names = []

# --- Plot 1: Line chart ---
fig1, ax1 = plt.subplots(figsize=(12, 7), dpi=150)

for i, f in enumerate(csv_files):
    test_idx = int(os.path.basename(f).replace('.csv', ''))
    df = pd.read_csv(f)
    
    episodes = df['episode'].values
    wins = df['win_count'].values
    
    win_rate = wins / (episodes + 1) * 100
    
    label = test_labels.get(test_idx, f'Test {test_idx}')
    color = colors[i % len(colors)]
    
    jitter = i * 0.3
    
    ax1.plot(episodes, win_rate + jitter, linewidth=1.8, color=color, alpha=0.85, label=f'T{test_idx}: {label}')
    
    final_rate = wins[-1] / (episodes[-1] + 1) * 100 if len(episodes) > 0 else 0
    final_rates.append(final_rate)
    test_names.append(f'T{test_idx}')

ax1.axhline(y=25, color='#888780', linestyle='--', linewidth=1, alpha=0.6, label='Random baseline (4-player, 25%)')
ax1.axhline(y=33.3, color='#B4B2A9', linestyle=':', linewidth=1, alpha=0.5, label='Random baseline (3-player, 33%)')
ax1.axhline(y=50, color='#D3D1C7', linestyle=':', linewidth=1, alpha=0.4, label='Random baseline (2-player, 50%)')
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Cumulative win rate (%)', fontsize=12)
ax1.legend(fontsize=8, loc='upper left', ncol=2, framealpha=0.9)
ax1.set_ylim(-2, 105)
ax1.set_xlim(left=0)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig('testingPlot_lines.png', bbox_inches='tight')
plt.close(fig1)
print("Saved testingPlot_lines.png")

# --- Plot 2: Bar chart ---
fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)

bar_colors = [colors[i % len(colors)] for i in range(len(final_rates))]
bars = ax2.bar(test_names, final_rates, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
ax2.axhline(y=25, color='#888780', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_ylabel('Final win rate (%)', fontsize=11)
ax2.set_ylim(0, 105)
ax2.grid(True, axis='y', alpha=0.3)

for bar, rate in zip(bars, final_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f'{rate:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
fig2.savefig('testingPlot_bars.png', bbox_inches='tight')
plt.close(fig2)
print("Saved testingPlot_bars.png")