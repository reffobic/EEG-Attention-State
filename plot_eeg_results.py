import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/refobic/Downloads/Project/EEG_KNN_Results.csv')

# Option A: using pivot_table
pivot_acc = df.pivot_table(
    index='Scenario',
    columns='Subject',
    values='Accuracy',
    aggfunc='max'
)

# Option B: group‐then‐pivot
# df_max = df.groupby(['Scenario','Subject'], as_index=False)['Accuracy'].max()
# pivot_acc = df_max.pivot(index='Scenario', columns='Subject', values='Accuracy')

# Bar chart
pivot_acc.T.plot(kind='bar', figsize=(10,6))
plt.ylabel('Accuracy')
plt.title('Test Accuracy by Scenario for Each Subject')
plt.xticks(rotation=0)
plt.legend(title='Scenario', bbox_to_anchor=(1.01,1), loc='upper left')
plt.tight_layout()
plt.show()

# Heatmap
fig, ax = plt.subplots(figsize=(8,5))
cax = ax.imshow(pivot_acc, aspect='auto', cmap='viridis')
ax.set_yticks(range(len(pivot_acc.index)))
ax.set_yticklabels(pivot_acc.index)
ax.set_xticks(range(len(pivot_acc.columns)))
ax.set_xticklabels(pivot_acc.columns)
plt.xlabel('Subject')
plt.title('Accuracy Heatmap (Scenario × Subject)')
fig.colorbar(cax, label='Accuracy')
plt.tight_layout()
plt.show()
