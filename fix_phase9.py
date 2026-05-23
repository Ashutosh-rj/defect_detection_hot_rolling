import os

with open('phase9_high_accuracy.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_main = False
for line in lines:
    if line.startswith('print("=" * 80)'):
        new_lines.append('if __name__ == "__main__":\n')
        in_main = True
    
    if in_main:
        if line == '\n':
            new_lines.append(line)
        else:
            new_lines.append('    ' + line)
    else:
        new_lines.append(line)

with open('phase9_high_accuracy.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Fixed phase9 multiprocessing guard.')
