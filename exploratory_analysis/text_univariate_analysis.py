import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os


data = pd.read_csv(os.getcwd().split('/exploratory_analysis')[0] + '/data/similar-staff-picks-challenge-clips_translation.csv')
# df = data[["category_names", "id"]].fillna('NA')
# df = df.groupby(['category_names']).count()
# df.to_csv(os.getcwd().split('/exploratory_analysis')[0] + '/data/similar-staff-picks-challenge-clips_categories.csv')


# Document lengths.
lens = [len(doc) for doc in list(data['title_en'].fillna(''))]

# Plot.
plt.rc('figure', figsize=(8,6))
plt.rc('font', size=14)
plt.rc('lines', linewidth=2)
# Histogram.
plt.hist(lens, bins=20)
plt.hold(True)
# Average length.
avg_len = sum(lens) / float(len(lens))
plt.axvline(avg_len, color='#e41a1c')
plt.hold(False)
plt.title('Histogram of Title Lengths.')
plt.xlabel('Length')
plt.text(100, 800, 'mean = %.2f' % avg_len)
plt.show()
#
# df = data[["title_language", "id"]].fillna('NA')
# df = df.groupby(['title_language']).count().reset_index()
# print(df)
#
# # Initialize the figure with a logarithmic x axis
# f, ax = plt.subplots(figsize=(7, 6))
#
#
# # Plot the orbital period with horizontal boxes
# sns.barplot(x="id", y="title_language", data=df)
#
# plt.show()
